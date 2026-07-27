"""process_state.json 读写工具（段级断点续传）。

被 setup_and_run.sh 调用。所有子命令用【退出码】表示结果，方便 bash 里 if 判断：
    0 = 是 / 成功        非 0 = 否 / 失败

  is-day-done    STATE TAG                    这一天是否已全部完成
  is-seg-done    STATE TAG SEG                这一段是否已完成或已判为残段跳过
  day-real-count STATE TAG                    打印这天状态为 done(有真实产出)的段数
  mark-seg       STATE TAG SEG STATUS [OUT]   记一段 (STATUS = done | skipped)
  mark-day       STATE TAG [OUT] [STATUS]     记一天完成 (STATUS = done | skipped)

写入是原子的（临时文件 + os.replace + fsync），中途断电不会留下半截 JSON。
state 文件损坏时会先备份成 <name>.bad[N] 再新建；备份失败则返回非零，
调用方（setup_and_run.sh）据此决定【不要删除原始数据】。

state 结构（旧格式只有天级 done，仍然兼容）：
{
  "GTACameraCapture_0722": {
    "done": true, "status": "done", "time": "...", "out": "seg0 seg1",
    "segs": {
      "0": {"status": "done",    "time": "...", "out": "E:/DATASET/rgb_depth_dataset_0722/seg0"},
      "6": {"status": "skipped", "time": "...", "out": "..."}
    }
  }
}
"""

import datetime
import json
import os
import sys


USAGE = __doc__


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load(path, repair=False):
    """读 state。repair=True 时遇到损坏文件会备份并新建（写入路径用）；
    repair=False 只是警告并当成空（查询路径用，绝不动用户的文件）。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError("顶层不是 JSON 对象")
        return data
    except Exception as exc:
        if not repair:
            print(f"!! 断点记录无法读取（{exc}），本次当成「没做过」处理: {path}",
                  file=sys.stderr)
            return {}
        bad = path + ".bad"
        n = 1
        while os.path.exists(bad):
            bad = "%s.bad%d" % (path, n)
            n += 1
        os.replace(path, bad)   # 失败就抛出去 -> 调用方拿到非零退出码 -> 不删原始
        print(f"!! 断点记录损坏（{exc}），已备份为 {bad}，将新建", file=sys.stderr)
        return {}


def save(path, data):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)       # 原子替换，不会出现半截文件


def seg_status(entry, seg):
    if entry.get("done") and not entry.get("segs"):
        return "done"           # 旧格式：整天完成，视作每段都完成
    return entry.get("segs", {}).get(str(seg), {}).get("status")


def run(argv):
    if len(argv) < 3:
        print(USAGE, file=sys.stderr)
        return 2
    cmd, state, tag = argv[0], argv[1], argv[2]

    if cmd == "is-day-done":
        return 0 if load(state).get(tag, {}).get("done") else 1

    if cmd == "is-seg-done":
        if len(argv) < 4:
            print(USAGE, file=sys.stderr)
            return 2
        status = seg_status(load(state).get(tag, {}), argv[3])
        return 0 if status in ("done", "skipped") else 1

    if cmd == "day-real-count":
        entry = load(state).get(tag, {})
        segs = entry.get("segs")
        if segs:
            print(sum(1 for v in segs.values() if v.get("status") == "done"))
        else:
            print(1 if entry.get("done") else 0)   # 旧格式整天 done 视作有产出
        return 0

    if cmd == "mark-seg":
        if len(argv) < 5:
            print(USAGE, file=sys.stderr)
            return 2
        seg, status = argv[3], argv[4]
        out = argv[5] if len(argv) > 5 else ""
        data = load(state, repair=True)
        entry = data.setdefault(tag, {})
        entry.setdefault("segs", {})[str(seg)] = {
            "status": status, "time": _now(), "out": out}
        save(state, data)
        print(f"  [{tag}] 段 {seg} 已记录: {status}  {out}")
        return 0

    if cmd == "mark-day":
        out = argv[3] if len(argv) > 3 else ""
        status = argv[4] if len(argv) > 4 else "done"
        data = load(state, repair=True)
        entry = data.setdefault(tag, {})
        entry["done"] = True
        entry["status"] = status
        entry["time"] = _now()
        entry["out"] = out.strip()
        save(state, data)
        print(f"  [{tag}] 整天已记录: {status}  {out.strip()}")
        return 0

    print(f"unknown command: {cmd}\n{USAGE}", file=sys.stderr)
    return 2


def main():
    try:
        return run(sys.argv[1:])
    except Exception as exc:                       # noqa: BLE001
        print(f"!! state_tool 失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
