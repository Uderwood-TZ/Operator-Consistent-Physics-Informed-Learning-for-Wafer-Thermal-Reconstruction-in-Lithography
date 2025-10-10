import numpy as np
import matplotlib.pyplot as plt
import matplotlib



# 加载数据
def load_data(file_path):
    data = np.loadtxt(file_path)
    x, y, T = data[:, 0], data[:, 1], data[:, 2]
    return x, y, T

# 计算误差（不依赖 sklearn）
def compute_error_metrics(T_true, T_pred):
    mse = np.mean((T_true - T_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(T_true - T_pred))
    max_error = np.max(np.abs(T_true - T_pred))
    return mse, rmse, mae, max_error

# 主程序
def main(true_file, pred_file):
    x_true, y_true, T_true = load_data(true_file)
    x_pred, y_pred, T_pred = load_data(pred_file)

    # 坐标对齐检查
    if not np.allclose(x_true, x_pred):
        print("x坐标不一致，最大差值：", np.max(np.abs(x_true - x_pred)))
    if not np.allclose(y_true, y_pred):
        print("y坐标不一致，最大差值：", np.max(np.abs(y_true - y_pred)))

    coords_true = np.vstack((x_true, y_true)).T
    coords_pred = np.vstack((x_pred, y_pred)).T

    # 尝试排序匹配
    if not np.allclose(coords_true, coords_pred):
        print("尝试根据坐标排序以匹配数据...")
        sorted_indices_true = np.lexsort((y_true, x_true))
        sorted_indices_pred = np.lexsort((y_pred, x_pred))

        coords_true = coords_true[sorted_indices_true]
        coords_pred = coords_pred[sorted_indices_pred]
        T_true = T_true[sorted_indices_true]
        T_pred = T_pred[sorted_indices_pred]

        if not np.allclose(coords_true, coords_pred):
            raise ValueError("排序后坐标仍不匹配，请检查数据文件")

    # 误差指标
    mse, rmse, mae, max_error = compute_error_metrics(T_true, T_pred)


    print(f"均方根误差 (RMSE): {rmse:.30f}")
    print(f"平均绝对误差 (MAE): {mae:.30f}")
    print(f"均方误差(MSE): {mse:.30f}")
    # 保存误差指标到文件
    with open('误差.txt', 'w', encoding='utf-8') as f:
        f.write("RMSE, MAE, MSE\n")
        f.write(f"{rmse:.30f}, {mae:.30f}, {mse:.30f}\n")

    # 可视化误差分布
    error = np.abs(T_true - T_pred)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(coords_true[:, 0], coords_true[:, 1], c=error, cmap='hot', s=20)
    plt.colorbar(sc, label='max_error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("max_error.png", dpi=300)
    plt.show()

# 示例调用（请确认文件名正确）
main('Real.txt', 'Pre.txt')










