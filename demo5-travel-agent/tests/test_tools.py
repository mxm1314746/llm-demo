"""
测试工具函数（10 个航空工具）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.tools import (
    search_flights, book_ticket, check_in, change_flight,
    cancel_ticket, select_seat, query_baggage, get_airport_info,
    query_policy, query_order, resolve_date, execute_tool, TOOL_DEFINITIONS,
)


class TestToolDefinitions:
    """工具 Schema 定义测试"""

    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 10

    def test_each_tool_has_required_fields(self):
        for t in TOOL_DEFINITIONS:
            assert t["type"] == "function"
            func = t["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert "properties" in func["parameters"]

    def test_search_flights_schema(self):
        schema = TOOL_DEFINITIONS[0]["function"]
        assert schema["name"] == "search_flights"
        assert "from" in schema["parameters"]["required"]
        assert "to" in schema["parameters"]["required"]
        assert "date" in schema["parameters"]["required"]

    def test_book_ticket_schema(self):
        schema = TOOL_DEFINITIONS[1]["function"]
        assert schema["name"] == "book_ticket"
        assert "flight_no" in schema["parameters"]["required"]
        assert "passenger" in schema["parameters"]["required"]
        assert "id_card" in schema["parameters"]["required"]

    def test_query_order_schema(self):
        schema = TOOL_DEFINITIONS[2]["function"]
        assert schema["name"] == "query_order"
        assert "order_no" in schema["parameters"]["required"]

    def test_check_in_schema(self):
        schema = TOOL_DEFINITIONS[3]["function"]
        assert schema["name"] == "check_in"

    def test_change_flight_schema(self):
        schema = TOOL_DEFINITIONS[4]["function"]
        assert schema["name"] == "change_flight"

    def test_cancel_ticket_schema(self):
        schema = TOOL_DEFINITIONS[5]["function"]
        assert schema["name"] == "cancel_ticket"

    def test_select_seat_schema(self):
        schema = TOOL_DEFINITIONS[6]["function"]
        assert schema["name"] == "select_seat"

    def test_query_baggage_schema(self):
        schema = TOOL_DEFINITIONS[7]["function"]
        assert schema["name"] == "query_baggage"

    def test_get_airport_info_schema(self):
        schema = TOOL_DEFINITIONS[8]["function"]
        assert schema["name"] == "get_airport_info"

    def test_query_policy_schema(self):
        schema = TOOL_DEFINITIONS[9]["function"]
        assert schema["name"] == "query_policy"


class TestResolveDate:
    """日期解析测试"""

    def test_relative_tomorrow(self):
        import datetime
        d, _ = resolve_date("明天")
        assert d == datetime.date.today() + datetime.timedelta(days=1)

    def test_iso_passthrough(self):
        import datetime
        d, _ = resolve_date("2030-01-15")
        assert d == datetime.date(2030, 1, 15)

    def test_month_day(self):
        d, _ = resolve_date("12月25日")
        assert (d.month, d.day) == (12, 25)


class TestSearchFlights:
    """search_flights 工具测试"""

    def test_normal_search(self):
        result = search_flights("北京", "上海", "明天")
        assert result.success is True
        assert "CZ3101" in result.message
        assert "4 个航班" in result.message
        assert len(result.data["flights"]) == 4

    def test_single_result(self):
        result = search_flights("北京", "成都", "明天")
        assert result.success is True
        assert "CZ4101" in result.message

    def test_no_flight_found(self):
        result = search_flights("北京", "拉萨", "明天")
        assert result.success is False

    def test_past_date_rejected(self):
        result = search_flights("北京", "上海", "2020-01-01")
        assert result.success is False


class TestBookTicket:
    """book_ticket 工具测试"""

    def test_book_success(self):
        result = book_ticket("CZ3101", "张三", "110101199001011234", "13800138000")
        assert result.success is True
        assert "订票成功" in result.message
        assert "CS-" in result.data["order_no"]

    def test_book_nonexistent_flight(self):
        result = book_ticket("CZ9999", "张三", "110101199001011234", "13800138000")
        assert result.success is False
        assert "不存在" in result.message

    def test_book_invalid_id(self):
        result = book_ticket("CZ3101", "张三", "123", "13800138000")
        assert result.success is False
        assert "身份证" in result.message


class TestQueryOrder:
    """query_order 工具测试"""

    def test_seeded_order_exists(self):
        result = query_order("CS-20260708-1234")
        assert result.success is True
        assert "状态" in result.message

    def test_nonexistent_order(self):
        result = query_order("CS-99999999-9999")
        assert result.success is False


class TestCheckIn:
    """check_in 工具测试"""

    def test_check_in_no_order(self):
        result = check_in("NONEXIST-ORDER", "张三")
        assert result.success is False


class TestChangeFlight:
    """change_flight 工具测试"""

    def test_change_nonexistent_order(self):
        result = change_flight("NONEXIST-ORDER", "CZ3103")
        assert result.success is False
        assert "不存在" in result.message


class TestCancelTicket:
    """cancel_ticket 工具测试"""

    def test_cancel_nonexistent_order(self):
        result = cancel_ticket("NONEXIST-ORDER")
        assert result.success is False
        assert "不存在" in result.message


class TestSelectSeat:
    """select_seat 工具测试"""

    def test_select_nonexistent_order(self):
        result = select_seat("NONEXIST-ORDER", "window")
        assert result.success is False
        assert "不存在" in result.message


class TestQueryBaggage:
    """query_baggage 工具测试"""

    def test_economy(self):
        result = query_baggage("", "经济舱")
        assert result.success is True
        assert "随身" in result.message
        assert "20kg" in result.message

    def test_first_class(self):
        result = query_baggage("", "头等舱")
        assert result.success is True
        assert "40kg" in result.message


class TestGetAirportInfo:
    """get_airport_info 工具测试"""

    def test_pek(self):
        result = get_airport_info("PEK")
        assert result.success is True
        assert "北京首都国际机场" in result.message

    def test_pvg(self):
        result = get_airport_info("PVG")
        assert result.success is True
        assert "上海浦东" in result.message

    def test_invalid_code(self):
        result = get_airport_info("XYZ")
        assert result.success is False


class TestQueryPolicy:
    """query_policy 工具测试"""

    def test_refund_policy(self):
        result = query_policy("退票手续费怎么算")
        assert result.success is True
        assert "手续费" in result.message

    def test_baggage_policy(self):
        result = query_policy("经济舱免费行李额")
        assert result.success is True

    def test_pet_policy(self):
        result = query_policy("宠物可以带上飞机吗")
        assert result.success is True
        assert "宠物" in result.message


class TestExecuteTool:
    """execute_tool 统一分发器测试"""

    def test_search_flights(self):
        result = execute_tool("search_flights", {"from": "北京", "to": "上海", "date": "明天"})
        assert "CZ3101" in result

    def test_unknown_tool(self):
        result = execute_tool("unknown_tool", {})
        assert "未知工具" in result

    def test_baggage(self):
        result = execute_tool("query_baggage", {})
        assert "行李" in result

    def test_airport(self):
        result = execute_tool("get_airport_info", {"airport_code": "PEK"})
        assert "北京首都" in result

    def test_policy(self):
        result = execute_tool("query_policy", {"question": "退票手续费"})
        assert "手续费" in result
