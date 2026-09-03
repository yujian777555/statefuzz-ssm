def join_values(values: list[str]) -> str:
    """按查询顺序连接精确值。"""
    if not values:
        raise ValueError("oracle至少需要一个值")
    if any("|" in value for value in values):
        raise ValueError("值中不得包含答案分隔符")
    return "|".join(values)
