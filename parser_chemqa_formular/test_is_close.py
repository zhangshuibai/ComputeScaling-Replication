from math import isclose

a = 340.0
b = 350.0

# 设置绝对容差
print(isclose(a, b, abs_tol=30,rel_tol=1e-6))  # 输出 True
print(isclose(a, b, abs_tol=2.9, rel_tol=1e-6))  # 输出 False（差值超出范围）