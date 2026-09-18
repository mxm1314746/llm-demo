"""记忆管线冒烟测试 — 用假 LLM 驱动 L0→L1→L2→L3 全流程（不消耗 API）"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory import store as mstore
from memory.manager import MemoryManager
from memory.pipeline import MemoryConfig


class FakeMsg:
    def __init__(self, content): self.message = type("M", (), {"content": content})
class FakeChoice:
    def __init__(self, content): self.choices = [FakeMsg(content)]
class FakeCompletions:
    def create(self, **kw):
        sysp = kw["messages"][0]["content"]
        if "记忆提炼器" in sysp:
            return FakeChoice(json.dumps({"summary": "用户咨询退票手续费规则", "tags": ["退票", "手续费"]}, ensure_ascii=False))
        if "长期记忆结构" in sysp:
            return FakeChoice("用户多次咨询退票手续费与退款到账时间")
        if "可复用技能" in sysp:
            return FakeChoice("【退票】适用：用户询问退票费。结论：按距起飞时间分档收费；退款3-7工作日到账。")
        return FakeChoice("ok")
class FakeChat:
    def __init__(self): self.completions = FakeCompletions()
class FakeClient:
    def __init__(self): self.chat = FakeChat()


def run():
    mstore.wipe_all()
    mgr = MemoryManager(FakeClient(), "fake-model",
                        MemoryConfig(episode_rounds=2, debounce_threshold=2, require_approval=True))
    sid = "test-sess"
    # 造 3 个 episode（每个 2 轮），tag 相同 → 应归并到同一 structure，第2次合并触发 skill(pending)
    for i in range(3):
        mgr.record_message(sid, "user", f"退票手续费怎么算？第{i}次", "private", "default")
        mgr.record_message(sid, "assistant", f"按距起飞时间分档，第{i}次回答", "private", "default")
        ep = mgr.close_episode(sid, force=True)
        assert ep is not None, "episode 未生成"

    st = mgr.stats()
    print("STATS:", st)
    assert st["L1_episodes"] == 3, st
    assert st["L2_structures"] == 1, f"应归并为1个structure: {st}"
    assert st["L3_skills"] == 1, f"应提取1个skill(防抖阈值2): {st}"
    assert st["L3_pending"] == 1, "skill 应处于待审批"
    assert st["skills_indexed"] == 0, "未审批不应入向量库"

    # 审批 → 入库
    pend = mgr.pending_skills()
    assert len(pend) == 1
    mgr.approve_skill(pend[0].id)
    st2 = mgr.stats()
    print("AFTER APPROVE:", st2)
    assert st2["L3_approved"] == 1
    assert st2["skills_indexed"] == 1, "审批后应写入 skills 向量库"

    # recall
    rec = mgr.recall("退票要多少钱", scope="private", owner="default")
    txt = mgr.format_recall(rec)
    print("RECALL:\n", txt)
    assert "退票" in txt

    # store_note 主动记忆
    ep = mgr.store_note("我只坐靠窗座位", ["座位偏好"], "private", "default")
    assert ep.tags == ["座位偏好"]
    print("STORE_NOTE ok:", ep.summary)

    print("\n✅ 全部断言通过")


if __name__ == "__main__":
    run()
