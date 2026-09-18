"""
工具定义 + 模拟实现 — 10个航空出行工具
核心能力:
  1. 动态日期解析（今天/明天/后天/N天后/周X/M月D日/YYYY-MM-DD），始终以真实当前时间为基准
  2. 航班数据按请求日期动态生成（航班号/时刻稳定，余票与价格随日期确定性变化）
  3. 订单持久化到 data/orders.json，含完整生命周期状态，支持 query_order 查询
"""
import json
import os
import re
import random
import hashlib
import datetime
import time
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ORDERS_FILE = os.path.join(DATA_DIR, "orders.json")


@dataclass
class ToolResult:
    """统一的工具返回格式"""
    success: bool
    data: dict = field(default_factory=dict)
    message: str = ""
    error: str = ""


# ╔══════════════════════════════════════════════════════════════╗
# ║  工具 Schema 定义（OpenAI Function Calling 格式）            ║
# ╚══════════════════════════════════════════════════════════════╝

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "搜索航班信息。根据出发地、目的地和日期查询可用航班，返回航班号、时间、价格、余票等信息。日期支持相对表述（今天/明天/后天）或具体日期。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "出发城市，如'北京'、'上海'、'广州'"},
                    "to": {"type": "string", "description": "目的地城市"},
                    "date": {"type": "string", "description": "出发日期。可传YYYY-MM-DD，也可传'今天'/'明天'/'后天'/'周六'/'9月20日'等，系统会自动解析为真实日期"},
                },
                "required": ["from", "to", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_ticket",
            "description": "预订机票。在用户确认航班后执行订票操作，需要旅客信息和航班号。订票前应先用search_flights查询航班。",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {"type": "string", "description": "航班号，如'CZ3101'"},
                    "passenger": {"type": "string", "description": "旅客姓名"},
                    "id_card": {"type": "string", "description": "身份证号"},
                    "phone": {"type": "string", "description": "联系电话"},
                    "date": {"type": "string", "description": "航班日期（可选，YYYY-MM-DD或相对日期）。缺省默认为明天"},
                },
                "required": ["flight_no", "passenger", "id_card", "phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询订单状态与详情。凭订单号返回航班、旅客、金额、当前状态（已出票/已值机/已选座/已改签/已退票）、座位、登机口等。用户问'我的订单'、'订单状态'、'值机了吗'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_no": {"type": "string", "description": "订单号/订票参考号，如'CS-20260708-1234'"},
                },
                "required": ["order_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_in",
            "description": "在线值机。凭订单号和旅客姓名办理值机手续，获取登机牌信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_no": {"type": "string", "description": "订单号/订票参考号"},
                    "passenger": {"type": "string", "description": "旅客姓名"},
                },
                "required": ["order_no", "passenger"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_flight",
            "description": "改签机票。将已有订单的航班变更为新的航班，会计算改签费用。需要订单号和新的航班号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_no": {"type": "string", "description": "原订单号"},
                    "new_flight_no": {"type": "string", "description": "新航班号"},
                },
                "required": ["order_no", "new_flight_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_ticket",
            "description": "退票。取消已有订单并根据距起飞时间计算退款金额。退票涉及手续费，需要用户确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_no": {"type": "string", "description": "要退票的订单号"},
                },
                "required": ["order_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_seat",
            "description": "选座。根据偏好为已订机票选择座位，支持窗口、过道、前排等偏好。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_no": {"type": "string", "description": "订单号"},
                    "preference": {
                        "type": "string",
                        "enum": ["window", "aisle", "front", "extra_legroom"],
                        "description": "座位偏好: window=靠窗, aisle=过道, front=前排, extra_legroom=腿部空间",
                    },
                },
                "required": ["order_no", "preference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_baggage",
            "description": "查询行李政策，包括免费行李额、超重费用、禁运物品等信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "route": {"type": "string", "description": "航线，如'北京-上海'，可选"},
                    "cabin_class": {
                        "type": "string",
                        "enum": ["经济舱", "公务舱", "头等舱"],
                        "description": "舱位等级，默认'经济舱'",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_airport_info",
            "description": "获取机场详细信息，包括航站楼、设施、交通方式、天气等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_code": {"type": "string", "description": "机场三字码，如'PEK'（北京）、'PVG'（上海）、'CAN'（广州）"},
                },
                "required": ["airport_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_policy",
            "description": "查询航空政策和规定，如退改签规则、行李规定、特殊旅客服务等。用于回答旅客关于政策的问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "要查询的政策问题，如'退票手续费怎么算'、'宠物可以带上飞机吗'"},
                },
                "required": ["question"],
            },
        },
    },
]


# ╔══════════════════════════════════════════════════════════════╗
# ║  数据加载 + 日期解析 + 航班动态生成                          ║
# ╚══════════════════════════════════════════════════════════════╝

def _load_json(filename: str) -> dict:
    """加载数据文件"""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _stable_int(seed: str, lo: int, hi: int) -> int:
    """基于字符串种子的确定性整数（同一日期+航班号每次结果一致）"""
    h = int(hashlib.md5(seed.encode("utf-8")).hexdigest(), 16)
    return lo + h % (hi - lo + 1)


_WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def resolve_date(date_str: str, now: datetime.datetime | None = None) -> tuple[datetime.date, str]:
    """
    将自然语言/数字日期解析为真实日期。
    返回 (date, 归一化标签)。支持:
      - 空 → 今天
      - 相对: 今天/明天/后天/大后天/N天后
      - 具体: YYYY-MM-DD, YYYY/MM/DD, YYYY年M月D日, M月D日/号
      - 星期: 周一..周日/周末/下周X
    解析失败时回退为今天。
    """
    now = now or datetime.datetime.now()
    today = now.date()
    s = (date_str or "").strip()
    if not s:
        return today, "今天"

    # 相对天数字典
    rel_map = {
        "今天": 0, "今日": 0, "当天": 0, "本日": 0,
        "明天": 1, "明日": 1, "明儿": 1,
        "后天": 2, "後天": 2, "明后天": 2,
        "大后天": 3, "三天后": 3,
    }
    for kw, delta in rel_map.items():
        if kw in s:
            d = today + timedelta(days=delta)
            return d, f"{kw}({d.isoformat()})"

    # N天后 / N天之后
    m = re.search(r"(\d+)\s*天(?:之)?后", s)
    if m:
        d = today + timedelta(days=int(m.group(1)))
        return d, f"{m.group(1)}天后({d.isoformat()})"

    # 完整日期 YYYY-MM-DD / YYYY/MM/DD / YYYY年M月D日
    m = re.search(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})", s)
    if m:
        try:
            d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d, d.isoformat()
        except ValueError:
            pass

    # M月D日/号（补当前年，若已过则顺延到明年）
    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", s)
    if m:
        mon, day = int(m.group(1)), int(m.group(2))
        for year in (today.year, today.year + 1):
            try:
                d = datetime.date(year, mon, day)
                if d >= today:
                    return d, d.isoformat()
            except ValueError:
                continue

    # 星期: 下周X / 周X / 星期X / 礼拜X / 周末
    wd_index = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    m = re.search(r"(下)?(?:周|星期|礼拜)\s*([一二三四五六日天])", s)
    if m:
        target = wd_index[m.group(2)]
        delta = (target - today.weekday()) % 7
        if m.group(1):  # 下周
            delta += 7 if delta != 0 else 7
        if delta == 0:
            delta = 7  # 已过去的今天本周同一天 → 下一周
        d = today + timedelta(days=delta)
        return d, f"{m.group(0)}({d.isoformat()})"
    if "周末" in s:
        delta = (5 - today.weekday()) % 7 or 7  # 最近的周六
        d = today + timedelta(days=delta)
        return d, f"周末({d.isoformat()})"

    # 纯数字 YYYYMMDD
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", s)
    if m:
        try:
            d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d, d.isoformat()
        except ValueError:
            pass

    return today, f"今天(未识别'{date_str}'，按今天处理)"


def _flight_templates() -> list[dict]:
    """航班时刻模板（与日期无关）"""
    return _load_json("flights.json")["flights"]


def _known_cities() -> list[str]:
    return _load_json("flights.json").get("cities", [])


def generate_flights(from_city: str, to_city: str, date: datetime.date,
                     now: datetime.datetime | None = None) -> list[dict]:
    """
    为指定日期动态生成航班：航班号/时刻来自模板保持稳定，
    余票与价格按 (航班号+日期) 确定性变化，周末上浮，贴近真实。
    """
    now = now or datetime.datetime.now()
    dstr = date.isoformat()
    weekend = date.weekday() >= 5
    out = []
    for t in _flight_templates():
        if t["from"] != from_city or t["to"] != to_city:
            continue
        # 若是今天，过滤掉已起飞的航班
        if date == now.date() and t["departure"] <= now.strftime("%H:%M"):
            continue
        price = t["price"]
        if weekend:
            price = int(price * 1.12)
        price += _stable_int(t["flight_no"] + dstr, -40, 40)
        price = max(300, (price // 10) * 10)
        seats = _stable_int(t["flight_no"] + dstr + "#seat", 2, 50)
        out.append({**t, "date": dstr, "price": price, "seats": seats})
    return out


def get_flight(flight_no: str, date: datetime.date, now: datetime.datetime | None = None) -> dict | None:
    """按航班号+日期取单个航班（动态生成）"""
    for t in _flight_templates():
        if t["flight_no"] == flight_no:
            flights = generate_flights(t["from"], t["to"], date, now)
            return next((f for f in flights if f["flight_no"] == flight_no), None)
    return None


# ╔══════════════════════════════════════════════════════════════╗
# ║  订单持久化存储                                              ║
# ╚══════════════════════════════════════════════════════════════╝

_bookings: dict = {}  # {order_no: booking_info}


def _mask_id(id_card: str) -> str:
    id_card = str(id_card or "")
    if len(id_card) <= 4:
        return id_card
    return "*" * (len(id_card) - 4) + id_card[-4:]


def _mask_phone(phone: str) -> str:
    phone = str(phone or "")
    if len(phone) < 7:
        return phone
    return phone[:3] + "****" + phone[-4:]


def _generate_order_no(now: datetime.datetime | None = None) -> str:
    now = now or datetime.datetime.now()
    return f"CS-{now.strftime('%Y%m%d')}-{random.randint(1000, 9999)}"


def _make_booking(order_no, flight, passenger, id_card, phone, status="已出票",
                  now: datetime.datetime | None = None) -> dict:
    now = now or datetime.datetime.now()
    dep_dt = datetime.datetime.fromisoformat(f"{flight['date']}T{flight['departure']}:00")
    return {
        "order_no": order_no,
        "flight_no": flight["flight_no"],
        "passenger": passenger,
        "id_card": _mask_id(id_card),
        "phone": _mask_phone(phone),
        "from": flight["from"],
        "to": flight["to"],
        "from_code": flight.get("from_code", ""),
        "to_code": flight.get("to_code", ""),
        "route": f"{flight['from']} → {flight['to']}",
        "date": flight["date"],
        "time": f"{flight['departure']}-{flight['arrival']}",
        "dep_datetime": dep_dt.isoformat(),
        "price": flight["price"],
        "cabin": flight.get("cabin", "经济舱"),
        "status": status,
        "created_at": now.isoformat(),
    }


def _seed_orders(now: datetime.datetime | None = None) -> dict:
    """预置两条演示订单，使 UI 示例与评测用例（CS-20260708-1234 等）可真实运行"""
    now = now or datetime.datetime.now()
    orders = {}
    demo_specs = [
        ("CS-20260708-1234", "CZ3101", "北京", "上海", "张三", "110101199001011234", "13800138000", 1),
        ("CS-20260708-5678", "CZ3201", "上海", "北京", "张三", "110101199001011234", "13900139000", 2),
    ]
    for order_no, fno, fc, tc, pax, idc, ph, day_offset in demo_specs:
        date = (now + timedelta(days=day_offset)).date()
        flights = generate_flights(fc, tc, date, now)
        flight = next((f for f in flights if f["flight_no"] == fno), None)
        if not flight and flights:
            flight = flights[0]
        if flight:
            orders[order_no] = _make_booking(order_no, flight, pax, idc, ph, "已出票", now)
    return orders


def _save_orders() -> None:
    try:
        with open(ORDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(_bookings, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _init_orders() -> None:
    global _bookings
    if os.path.exists(ORDERS_FILE):
        try:
            with open(ORDERS_FILE, "r", encoding="utf-8") as f:
                _bookings = json.load(f)
                return
        except (OSError, json.JSONDecodeError):
            pass
    _bookings = _seed_orders()
    _save_orders()


_init_orders()


# ╔══════════════════════════════════════════════════════════════╗
# ║  工具执行函数                                               ║
# ╚══════════════════════════════════════════════════════════════╝

def search_flights(from_city: str, to_city: str, date: str) -> ToolResult:
    """搜索航班（动态日期 + 动态生成）"""
    now = datetime.datetime.now()
    from_city = (from_city or "").replace("市", "").strip()
    to_city = (to_city or "").replace("市", "").strip()
    resolved, label = resolve_date(date, now)

    known = _known_cities()
    if from_city not in known or to_city not in known:
        missing = [c for c in (from_city, to_city) if c not in known]
        return ToolResult(
            success=False,
            message=f"暂不支持城市 {'、'.join(missing)}。当前覆盖城市: {'、'.join(known)}。",
        )

    if resolved < now.date():
        return ToolResult(
            success=False,
            message=f"⚠️ 查询日期 {resolved.isoformat()}（{label}）已过期。今天是 {now.date().isoformat()}，请查询今天及以后的日期。",
        )

    results = generate_flights(from_city, to_city, resolved, now)
    if not results:
        hint = f"（今天已无剩余班次）" if resolved == now.date() else ""
        return ToolResult(
            success=False,
            message=f"{resolved.isoformat()}（{label}）从{from_city}到{to_city}暂无航班{hint}。建议更换日期或中转。",
        )

    weekday = _WEEKDAY_CN[resolved.weekday()]
    lines = [f"找到 {len(results)} 个航班 ({from_city} → {to_city}, {resolved.isoformat()} {weekday} · {label}):\n"]
    for f in results:
        lines.append(
            f"  ✈ {f['flight_no']} | {f['departure']}-{f['arrival']} | "
            f"{f['cabin']} | ¥{f['price']} | 余票{f['seats']}张"
        )
    return ToolResult(
        success=True,
        data={"date": resolved.isoformat(), "weekday": weekday, "flights": results},
        message="\n".join(lines),
    )


def book_ticket(flight_no: str, passenger: str, id_card: str, phone: str,
                date: str = "") -> ToolResult:
    """预订机票"""
    now = datetime.datetime.now()
    # 缺省日期 → 明天
    if not date:
        date = (now + timedelta(days=1)).date().isoformat()
    resolved, label = resolve_date(date, now)

    if len(str(id_card)) not in (15, 18):
        return ToolResult(success=False, message=f"身份证号格式不正确（{len(str(id_card))}位），应为15或18位，请核对后重试。")
    if not re.fullmatch(r"1[3-9]\d{9}", str(phone)):
        return ToolResult(success=False, message="手机号格式不正确，请输入11位有效手机号。")

    if resolved < now.date():
        return ToolResult(success=False, message=f"⚠️ 日期 {resolved.isoformat()} 已过期，无法订票。")

    flight = get_flight(flight_no, resolved, now)
    if not flight:
        return ToolResult(success=False, message=f"航班 {flight_no} 在 {resolved.isoformat()} 不存在，请检查航班号或日期。")
    if flight["seats"] <= 0:
        return ToolResult(success=False, message=f"航班 {flight_no} 已无余票。")

    order_no = _generate_order_no(now)
    booking = _make_booking(order_no, flight, passenger, id_card, phone, "已出票", now)
    _bookings[order_no] = booking
    _save_orders()

    weekday = _WEEKDAY_CN[resolved.weekday()]
    return ToolResult(
        success=True,
        data=booking,
        message=(
            f"🎫 订票成功！\n"
            f"订单号: {order_no}\n"
            f"航班: {flight_no} | {flight['from']} → {flight['to']}\n"
            f"日期: {flight['date']}（{weekday}）| {flight['departure']}-{flight['arrival']}\n"
            f"旅客: {passenger} | 金额: ¥{flight['price']}\n"
            f"状态: 已出票\n"
            f"可凭订单号查询状态、办理值机或退改签。"
        ),
    )


def query_order(order_no: str) -> ToolResult:
    """查询订单状态与详情"""
    booking = _bookings.get(order_no)
    if not booking:
        return ToolResult(
            success=False,
            message=f"未找到订单 {order_no}。请核对订单号（格式如 CS-YYYYMMDD-XXXX）。",
        )
    now = datetime.datetime.now()
    dep = datetime.datetime.fromisoformat(booking["dep_datetime"])
    hours_to_dep = (dep - now).total_seconds() / 3600
    if hours_to_dep < 0:
        time_hint = "航班已起飞"
    else:
        time_hint = f"距起飞约 {int(hours_to_dep)} 小时"

    extra = []
    if booking.get("seat"):
        extra.append(f"座位: {booking['seat']}")
    if booking.get("boarding_gate"):
        extra.append(f"登机口: {booking['boarding_gate']}")
    if booking.get("checked_in"):
        extra.append("值机: 已完成")
    if booking.get("old_flight_no"):
        extra.append(f"原航班: {booking['old_flight_no']} ({booking.get('old_time','')})")
    extra_str = ("\n" + "\n".join(extra)) if extra else ""

    return ToolResult(
        success=True,
        data=booking,
        message=(
            f"📦 订单 {order_no}\n"
            f"状态: {booking['status']}\n"
            f"航班: {booking['flight_no']} | {booking['route']}\n"
            f"日期: {booking['date']} | {booking['time']}（{time_hint}）\n"
            f"旅客: {booking['passenger']} | 金额: ¥{booking['price']}"
            f"{extra_str}"
        ),
    )


def check_in(order_no: str, passenger: str) -> ToolResult:
    """在线值机"""
    booking = _bookings.get(order_no)
    if not booking:
        return ToolResult(success=False, message=f"订单 {order_no} 不存在，请检查订单号。")
    if booking["passenger"] != passenger:
        return ToolResult(success=False, message=f"旅客姓名与订单不符（订单登记为 {booking['passenger']}）。")
    if booking["status"] == "已退票":
        return ToolResult(success=False, message="该订单已退票，无法值机。")

    now = datetime.datetime.now()
    dep = datetime.datetime.fromisoformat(booking["dep_datetime"])
    hours_to_dep = (dep - now).total_seconds() / 3600
    if hours_to_dep < 0:
        return ToolResult(success=False, message=f"航班已于 {booking['date']} {booking['time'].split('-')[0]} 起飞，无法办理值机。")
    if hours_to_dep > 48:
        return ToolResult(success=False, message=f"值机尚未开放。网上值机一般在航班起飞前48小时内开放，当前距起飞约 {int(hours_to_dep)} 小时。")

    if booking.get("checked_in"):
        return ToolResult(
            success=True, data=booking,
            message=f"您已完成值机，无需重复操作。\n座位: {booking.get('seat')} | 登机口: {booking.get('boarding_gate')}",
        )

    gates = [f"A{i}" for i in range(10, 30)] + [f"B{i}" for i in range(10, 30)]
    seat = booking.get("seat") or f"{random.randint(30, 60)}{random.choice('ABCDEF')}"

    booking["checked_in"] = True
    booking["status"] = "已值机"
    booking["check_in_time"] = now.isoformat()
    booking["boarding_gate"] = random.choice(gates)
    booking["boarding_time"] = booking["time"].split("-")[0]
    booking["seat"] = seat

    airports = _load_json("airports.json")
    airport_info = ""
    info = airports.get(booking.get("from_code", ""))
    if info:
        airport_info = f"\n出发航站楼: {random.choice(info['terminals'])}"

    _save_orders()
    return ToolResult(
        success=True,
        data=booking,
        message=(
            f"✅ 值机成功！\n"
            f"旅客: {passenger}\n"
            f"航班: {booking['flight_no']} | {booking['route']}\n"
            f"日期: {booking['date']} | {booking['time']}\n"
            f"座位: {seat}\n"
            f"登机口: {booking['boarding_gate']}{airport_info}\n"
            f"请提前45分钟到达登机口。"
        ),
    )


def change_flight(order_no: str, new_flight_no: str) -> ToolResult:
    """改签"""
    booking = _bookings.get(order_no)
    if not booking:
        return ToolResult(success=False, message=f"订单 {order_no} 不存在。")
    if booking["status"] == "已退票":
        return ToolResult(success=False, message="该订单已退票，无法改签。")
    if booking["status"] == "已改签":
        return ToolResult(success=False, message="该订单已改签过，每个订单仅限改签一次。")

    now = datetime.datetime.now()
    dep = datetime.datetime.fromisoformat(booking["dep_datetime"])
    if (dep - now).total_seconds() / 3600 < 2:
        return ToolResult(success=False, message="距起飞不足2小时，无法在线改签，请联系柜台或客服95539。")

    # 改签目标日期：默认沿用原订单日期
    target_date = datetime.date.fromisoformat(booking["date"])
    new_flight = get_flight(new_flight_no, target_date, now)
    if not new_flight:
        return ToolResult(success=False, message=f"航班 {new_flight_no} 在 {target_date.isoformat()} 不存在。")
    if new_flight["seats"] <= 0:
        return ToolResult(success=False, message=f"航班 {new_flight_no} 已无余票。")

    change_fee = int(booking["price"] * 0.15)
    price_diff = new_flight["price"] - booking["price"]
    total_fee = change_fee + max(0, price_diff)

    booking["old_flight_no"] = booking["flight_no"]
    booking["old_time"] = booking["time"]
    booking["flight_no"] = new_flight_no
    booking["time"] = f"{new_flight['departure']}-{new_flight['arrival']}"
    booking["route"] = f"{new_flight['from']} → {new_flight['to']}"
    booking["from_code"] = new_flight.get("from_code", booking.get("from_code", ""))
    booking["dep_datetime"] = datetime.datetime.fromisoformat(
        f"{new_flight['date']}T{new_flight['departure']}:00"
    ).isoformat()
    booking["price"] = new_flight["price"]
    booking["status"] = "已改签"
    booking["checked_in"] = False  # 改签后需重新值机
    booking.pop("boarding_gate", None)
    booking["changed_at"] = now.isoformat()
    _save_orders()

    return ToolResult(
        success=True,
        data=booking,
        message=(
            f"✅ 改签成功！\n"
            f"原航班: {booking['old_flight_no']} ({booking['old_time']})\n"
            f"新航班: {new_flight_no} ({new_flight['departure']}-{new_flight['arrival']}, {new_flight['date']})\n"
            f"改签费: ¥{change_fee} | 差价: ¥{max(0, price_diff)}\n"
            f"合计: ¥{total_fee}\n"
            f"提示: 改签后需重新办理值机。"
        ),
    )


def cancel_ticket(order_no: str) -> ToolResult:
    """退票（手续费按距起飞真实时间计算）"""
    booking = _bookings.get(order_no)
    if not booking:
        return ToolResult(success=False, message=f"订单 {order_no} 不存在。")
    if booking["status"] == "已退票":
        return ToolResult(success=False, message="该订单已退票。")

    now = datetime.datetime.now()
    dep = datetime.datetime.fromisoformat(booking["dep_datetime"])
    hours_until_flight = (dep - now).total_seconds() / 3600

    if hours_until_flight < 0:
        return ToolResult(success=False, message="航班已起飞，无法在线退票，请联系客服95539。")
    if hours_until_flight > 24:
        refund_rate = 0.90
    elif hours_until_flight > 2:
        refund_rate = 0.80
    else:
        refund_rate = 0.50

    refund = int(booking["price"] * refund_rate)
    fee = booking["price"] - refund

    booking["status"] = "已退票"
    booking["cancelled_at"] = now.isoformat()
    booking["refund"] = refund
    _save_orders()

    return ToolResult(
        success=True,
        data=booking,
        message=(
            f"✅ 退票成功！\n"
            f"订单: {order_no}\n"
            f"原票价: ¥{booking['price']}\n"
            f"手续费: ¥{fee} ({int((1 - refund_rate) * 100)}%，按距起飞{int(hours_until_flight)}小时计算)\n"
            f"退款金额: ¥{refund}\n"
            f"退款将在3-7个工作日内到账。"
        ),
    )


def select_seat(order_no: str, preference: str) -> ToolResult:
    """选座"""
    booking = _bookings.get(order_no)
    if not booking:
        return ToolResult(success=False, message=f"订单 {order_no} 不存在。")
    if booking["status"] == "已退票":
        return ToolResult(success=False, message="该订单已退票，无法选座。")

    seat_maps = {
        "window": ["{}A", "{}F"],
        "aisle": ["{}C", "{}D"],
        "front": ["{}A", "{}B", "{}C", "{}D", "{}E", "{}F"],
        "extra_legroom": ["{}A", "{}C", "{}D", "{}F"],
    }
    pattern = random.choice(seat_maps.get(preference, ["{}A"]))
    row = random.randint(8, 15) if preference == "front" else random.randint(30, 55)
    seat = pattern.format(row)

    pref_names = {"window": "靠窗", "aisle": "过道", "front": "前排", "extra_legroom": "腿部大空间"}
    booking["seat"] = seat
    booking["seat_preference"] = preference
    if booking["status"] == "已出票":
        booking["status"] = "已选座"
    _save_orders()

    return ToolResult(
        success=True,
        data={"seat": seat, "preference": preference},
        message=(
            f"💺 选座成功！\n"
            f"偏好: {pref_names.get(preference, preference)}\n"
            f"座位: {seat}（第{row}排）\n"
            f"航班: {booking['flight_no']} | {booking['route']} | {booking['date']}"
        ),
    )


def query_baggage(route: str = "", cabin_class: str = "经济舱") -> ToolResult:
    """查询行李政策"""
    free_allowances = {
        "经济舱": "随身1件(5kg) + 托运20kg",
        "公务舱": "随身2件(10kg) + 托运30kg",
        "头等舱": "随身2件(15kg) + 托运40kg",
    }
    free = free_allowances.get(cabin_class, free_allowances["经济舱"])
    route_str = f"（{route}航线）" if route else ""

    return ToolResult(
        success=True,
        message=(
            f"🧳 行李政策{route_str} | {cabin_class}:\n"
            f"免费额度: {free}\n"
            f"超重费: 全票价的 1.5%/kg\n"
            f"禁运品: 锂电池、充电宝(≤100Wh)、打火机、易燃易爆品\n"
            f"液体限制: 随身单瓶≤100ml，总量≤1000ml（需装入透明袋）\n"
            f"特殊行李: 运动器材、乐器可托运，需特殊包装"
        ),
    )


def get_airport_info(airport_code: str) -> ToolResult:
    """获取机场信息"""
    airports = _load_json("airports.json")
    # 支持城市名→代码
    code = (airport_code or "").strip().upper()
    if code not in airports:
        for c, info in airports.items():
            if info["city"] == airport_code or info["name"].startswith(airport_code):
                code = c
                break
    info = airports.get(code)
    if not info:
        return ToolResult(success=False, message=f"未找到机场 {airport_code} 的信息。可用: PEK(北京), PVG(上海), CAN(广州), CTU(成都), SZX(深圳)")

    return ToolResult(
        success=True,
        data=info,
        message=(
            f"🏢 {info['name']} ({code})\n"
            f"城市: {info['city']}\n"
            f"航站楼: {', '.join(info['terminals'])}\n"
            f"设施: {', '.join(info['facilities'])}\n"
            f"交通: {', '.join(info['transportation'])}\n"
            f"天气: {info['weather']}\n"
            f"服务热线: {info['hotline']}"
        ),
    )


def query_policy(question: str) -> ToolResult:
    """查询政策（关键词匹配 + JSON 知识库）"""
    policies = _load_json("policies.json")
    question_lower = question.lower()

    best_match = None
    best_score = 0
    for p in policies["policies"]:
        q_words = set(question)
        a_words = set(p["question"] + p["answer"])
        common = q_words & a_words
        score = len(common) / max(len(q_words), 1)

        categories = {
            "行李": ["行李", "托运", "携带", "液体", "充电宝", "宠物"],
            "退改签": ["退票", "改签", "退改", "手续费", "延误", "退款"],
            "值机": ["值机", "选座", "登机", "座位"],
            "证件": ["身份证", "证件", "护照"],
            "特殊旅客": ["老人", "儿童", "孕妇", "轮椅", "药品", "婴儿"],
            "会员": ["会员", "明珠", "积分", "里程"],
            "航班": ["航班", "动态", "舱位", "头等舱"],
        }
        for cat, keywords in categories.items():
            if any(kw in question for kw in keywords) and p["category"] == cat:
                score += 0.3

        if score > best_score:
            best_score = score
            best_match = p

    if best_match and best_score > 0.1:
        return ToolResult(
            success=True,
            data=best_match,
            message=f"📋 **{best_match['question']}**\n\n{best_match['answer']}",
        )
    else:
        return ToolResult(
            success=False,
            message=f"未找到与「{question}」相关的政策信息。建议联系南航客服 95539 获取帮助。",
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  工具分发器                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def execute_tool(name: str, args: dict) -> str:
    """工具分发器 — 根据工具名路由到对应函数，返回字符串给 LLM"""
    time.sleep(0.3)  # 模拟网络延迟

    try:
        if name == "search_flights":
            result = search_flights(args.get("from", ""), args.get("to", ""), args.get("date", ""))
        elif name == "book_ticket":
            result = book_ticket(
                args.get("flight_no", ""), args.get("passenger", ""),
                args.get("id_card", ""), args.get("phone", ""), args.get("date", ""),
            )
        elif name == "query_order":
            result = query_order(args.get("order_no", ""))
        elif name == "check_in":
            result = check_in(args.get("order_no", ""), args.get("passenger", ""))
        elif name == "change_flight":
            result = change_flight(args.get("order_no", ""), args.get("new_flight_no", ""))
        elif name == "cancel_ticket":
            result = cancel_ticket(args.get("order_no", ""))
        elif name == "select_seat":
            result = select_seat(args.get("order_no", ""), args.get("preference", ""))
        elif name == "query_baggage":
            result = query_baggage(args.get("route", ""), args.get("cabin_class", "经济舱"))
        elif name == "get_airport_info":
            result = get_airport_info(args.get("airport_code", ""))
        elif name == "query_policy":
            result = query_policy(args.get("question", ""))
        else:
            return f"❌ 未知工具: {name}"

        return result.message if result.success else f"⚠️ {result.message}"
    except Exception as e:
        return f"❌ 工具执行异常 [{name}]: {e}"
