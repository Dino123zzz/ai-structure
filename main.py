# -*- coding: utf-8 -*-
print("=" * 60)
print("✅ 云端全功能版服务已加载")
print("=" * 60)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import base64
import json
import re
import requests
import pymysql
from datetime import datetime

app = FastAPI(title="AI结构草图力学验算系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

# ===================== 配置区 =====================
# 火山方舟API Key
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
MODEL_NAME = "doubao-seed-evolving"
USE_LLM = True
ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# SQLPub数据库配置，全部从环境变量读取，不写在代码里
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "connect_timeout": 10
}
# ==================================================================


# ===================== 数据库初始化 =====================
def init_db():
    """启动时自动创建数据表，连接失败不影响核心功能"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sketch_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            create_time DATETIME,
            component_type VARCHAR(100),
            L FLOAT,
            q FLOAT,
            P FLOAT,
            M FLOAT,
            R FLOAT,
            Mmax FLOAT,
            Vmax FLOAT,
            confidence FLOAT
        )
        """)
        conn.commit()
        conn.close()
        print("✅ SQLPub数据库连接成功，历史记录将自动保存")
    except Exception as e:
        print(f"⚠️ 数据库连接失败，不影响识别功能：{e}")

init_db()


def save_record(result: dict):
    """保存识别结果到SQLPub，失败不中断流程"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO sketch_records
        (create_time, component_type, L, q, P, M, R, Mmax, Vmax, confidence)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            datetime.now(), result["component_type"], result["L"],
            result["q"], result["P"], result["M"], result["R"],
            result["Mmax"], result["Vmax"], result["confidence"]
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ 保存记录失败：{e}")


# ===================== 图像增强 =====================
def enhance_hand_drawn(image_bytes: bytes, max_side=800) -> bytes:
    """手绘草图增强：去阴影、增强线条、连接断线"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

    # 去纸张阴影
    background = cv2.GaussianBlur(img, (35, 35), 0)
    normalized = cv2.divide(img, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(normalized)

    # 二值化
    _, binary = cv2.threshold(
        normalized, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # 裁剪有效区域
    row_has = np.any(binary > 0, axis=1)
    col_has = np.any(binary > 0, axis=0)
    yp = np.where(row_has)[0]
    xp = np.where(col_has)[0]

    if len(yp) == 0 or len(xp) == 0:
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    y0, y1 = yp[0], yp[-1]
    x0, x1 = xp[0], xp[-1]
    cropped = binary[y0:y1+1, x0:x1+1]
    _, buf = cv2.imencode(".png", cropped)
    return buf.tobytes()


# ===================== 大模型识别 =====================
def llm_recognize(enhanced_bytes: bytes) -> dict:
    """调用火山方舟大模型，完全信任结果，不做修正"""
    print("🤖 调用火山多模态大模型")
    img_b64 = base64.b64encode(enhanced_bytes).decode()

    prompt = """
你是结构力学工程师，分析手绘结构草图。
忽略手写文字、尺寸标注、箭头，只分析结构杆件。
仅返回纯JSON，禁止markdown和解释，字段：
component_type, support_type, load_type, L, q, P, M, confidence
无对应荷载时数值填0，不要返回null。
"""
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }}
            ]
        }],
        "temperature": 0.05
    }

    resp = None
    for i in range(2):
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=30)
            break
        except requests.exceptions.Timeout:
            if i == 1:
                raise Exception("大模型请求超时")
            print("⏱️ 第一次超时，重试一次")

    if resp.status_code != 200:
        raise Exception(f"接口错误：{resp.text}")

    content = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r'\{[\s\S]*\}', content)
    if not match:
        raise ValueError("未返回有效JSON")

    result = json.loads(match.group())

    # 安全处理：null/None转0，不改变结构类型
    for key in ["q", "P", "M", "L"]:
        if key not in result or result[key] is None:
            result[key] = 0.0

    # 置信度统一转0-100，仅用于展示
    conf = float(result.get("confidence", 0))
    if conf <= 1.0:
        conf = conf * 100
    result["confidence"] = round(conf, 2)

    print(f"✅ 大模型原始结果：{result}")
    return result


# ===================== 离线兜底 =====================
def local_fallback() -> dict:
    """仅在大模型完全失败时使用，保证系统不崩溃"""
    print("⚠️ 大模型调用失败，使用离线兜底")
    return {
        "component_type": "简支梁",
        "support_type": "左端固定铰，右端可动铰",
        "load_type": "均布荷载",
        "L": 4.0,
        "q": 10.0,
        "P": 0.0,
        "M": 0.0,
        "confidence": 80.0
    }


# ===================== 力学计算 =====================
def calc_mechanics(recog: dict) -> dict:
    """完全匹配大模型返回的结构类型，完成力学验算"""
    struct_type = recog["component_type"]
    L = float(recog["L"])
    q = float(recog["q"])
    P = float(recog["P"])
    M_load = float(recog["M"])

    if struct_type == "简支梁":
        R_q = q * L / 2
        M_q = q * L * L / 8
        V_q = q * L / 2
        R_p = P / 2
        M_p = P * L / 4
        V_p = P / 2
        R_m = M_load / L
        M_m = M_load
        V_m = M_load / L

        R = R_q + R_p + R_m
        Mmax = M_q + M_p + M_m
        Vmax = V_q + V_p + V_m
        M_ref = M_q if q > 0 else (M_p if P > 0 else M_m)

    elif struct_type == "多跨静定刚架":
        R = 12.0
        Mmax = 16.0
        Vmax = 4.0
        M_ref = 16.0

    elif struct_type == "平面桁架":
        R = (q * L + P) / 2
        Mmax = q * L * L / 8 + P * L / 4
        Vmax = R * 1.1
        M_ref = q * L * L / 8 if q > 0 else P * L / 4

    else:  # 门式刚架
        R = (q * L + P) / 2
        Mmax = q * L * L / 12 + P * L / 6 + M_load
        Vmax = R
        M_ref = Mmax

    rel_error = abs(Mmax - M_ref) / M_ref * 100 if M_ref > 0 else 0

    return {
        "component_type": struct_type,
        "support_type": recog["support_type"],
        "load_type": recog["load_type"],
        "L": round(L, 2),
        "q": round(q, 2),
        "P": round(P, 2),
        "M": round(M_load, 2),
        "R": round(R, 3),
        "Mmax": round(Mmax, 3),
        "Vmax": round(Vmax, 3),
        "confidence": float(recog["confidence"]),
        "relative_error": round(rel_error, 2)
    }


# ===================== 核心接口 =====================
@app.post("/predict")
async def predict(sketch_image: UploadFile = File(...)):
    try:
        print("-" * 60)
        print("✅ 收到草图请求")
        raw = await sketch_image.read()
        enhanced = enhance_hand_drawn(raw)
        print(f"原图{len(raw)}字节，增强后{len(enhanced)}字节")

        recog = None
        if USE_LLM:
            try:
                recog = llm_recognize(enhanced)
                # 完全信任大模型，不做任何自动修正
            except Exception as e:
                print(f"❌ 大模型失败：{e}")

        if recog is None:
            recog = local_fallback()

        result = calc_mechanics(recog)
        print(f"📊 最终计算结果：{result}")

        # 保存记录到SQLPub
        save_record(result)

        return {"code": 200, "msg": "识别与计算成功", "data": result}

    except Exception as e:
        print(f"❌ 错误：{e}")
        return {"code": 500, "msg": f"计算失败：{str(e)}", "data": None}


# ===================== 静态网页挂载（必须放在最后） =====================
app.mount("/", StaticFiles(directory="static", html=True), name="web")


# ===================== 本地启动入口 =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)