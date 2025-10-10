% 清空环境
clear; clc; close all;

%% ========== 参数定义 ==========
k = 159;                % 热导率 (W/m·K)
h = 50;                 % 对流换热系数 (W/m²·K)
T_inf = 800;            % 环境温度 (K)
Q = 2000;               % 内部热源 (W/m³)
R = 0.15;               % 圆形区域半径 (m)

%% ========== 创建 PDE 模型 ==========
thermalModel = createpde('thermal','steadystate');

%% ========== 几何建模 ==========
gd = [1; 0; 0; R];  % 圆：类型=1，中心(0,0)，半径R
ns = char('C1'); ns = ns';
sf = 'C1';
g = decsg(gd,sf,ns);
geometryFromEdges(thermalModel,g);

%% ========== 材料属性和源项 ==========
thermalProperties(thermalModel, 'ThermalConductivity', k);
internalHeatSource(thermalModel, Q);

%% ========== 边界条件 ==========
thermalBC(thermalModel, ...
          'Edge', 1:thermalModel.Geometry.NumEdges, ...
          'ConvectionCoefficient', h, ...
          'AmbientTemperature', T_inf);

%% ========== 网格生成 ==========
Hmax = R / 30;  % 控制网格精度
generateMesh(thermalModel, 'Hmax', Hmax);

%% ========== 网格示意图（仅网格线） ==========
figure('Color','w');
pdeplot(thermalModel, 'Mesh', 'on');     % 只画网格线
axis equal tight;
xlabel('x (m)'); ylabel('y (m)');
title(sprintf('FEM Mesh (H_{max}=%.4g m, N_{nodes}=%d, N_{elem}=%d)', ...
      Hmax, size(thermalModel.Mesh.Nodes,2), size(thermalModel.Mesh.Elements,2)));
box on;
exportgraphics(gcf, 'mesh_schematic.png', 'Resolution', 300);
fprintf('网格示意图已保存为 mesh_schematic.png\n');


%% ========== 求解 PDE ==========
thermalResult = solve(thermalModel);
T = thermalResult.Temperature;
nodes = thermalModel.Mesh.Nodes;

%% ========== 可视化结果 ==========
figure;
pdeplot(thermalModel, ...
        'XYData', T, ...
        'ColorMap', 'jet', ...
        'Mesh', 'off', ...
        'FaceAlpha', 1);
axis equal;
xlabel('x (m)');
ylabel('y (m)');

cb = colorbar;
cb.Label.String = 'Temperature (K)';
tickVals = get(cb, 'Ticks');
tickLabels = arrayfun(@(x) sprintf('%.4f', x), tickVals, 'UniformOutput', false);
set(cb, 'TickLabels', tickLabels);
exportgraphics(gcf, 'temperature_distribution.png', 'Resolution', 300);
fprintf('温度分布图已保存为 temperature_distribution.png\n');

%% ========== 插值到规则网格 + 去除 NaN ==========
% FEM 点
x = nodes(1,:)';
y = nodes(2,:)';
T_vals = T;

% 构建规则网格（步长你可自定义）
dx = 0.002;
dy = 0.002;
[Xq, Yq] = meshgrid(-R:dx:R, -R:dy:R);

% 筛选出圆形区域内部点 (用于插值)
mask = Xq.^2 + Yq.^2 <= R^2;
Xq_valid = Xq(mask);
Yq_valid = Yq(mask);

% 使用 scatteredInterpolant 插值
F = scatteredInterpolant(x, y, T_vals, 'linear', 'linear');
Tq_valid = F(Xq_valid, Yq_valid);

%% ========== 导出温度数据 ==========
outputMatrix_uniform = [Xq_valid, Yq_valid, Tq_valid];
outputFile = 'Real.txt';

fid = fopen(outputFile, 'w');
fprintf(fid, 'x(m)\t\ty(m)\t\tTemperature(K)\n');
fprintf(fid, '%.6f\t%.6f\t%.4f\n', outputMatrix_uniform');
fclose(fid);

fprintf('统一规则网格温度数据已导出至：%s\n', outputFile);

%% ========== 输出圆心温度 ==========
[~, centerNodeIdx] = min(vecnorm(nodes - [0; 0]));
T_center = T(centerNodeIdx);
fprintf('圆心温度为：%.4f K\n', T_center);

% FEM节点和温度
x_fem = nodes(1,:)';
y_fem = nodes(2,:)';
T_fem = T;

% 读取对比点坐标文件，或者直接定义：
% comparePoints = [0.000376 0.147744; 0.000376 0.148496; ...];
comparePoints = load('cc.txt'); % 如果你有这个文件

x_cmp = comparePoints(:,1);
y_cmp = comparePoints(:,2);

% 插值
F = scatteredInterpolant(x_fem, y_fem, T_fem, 'linear', 'none');
T_cmp = F(x_cmp, y_cmp);

% 导出结果
outputMatrix_cmp = [x_cmp, y_cmp, T_cmp];
outputFile_cmp = 'temperature_compare_points.txt';

fid = fopen(outputFile_cmp, 'w');
fprintf(fid, 'x(m)\t\ty(m)\t\tTemperature(K)\n');
fprintf(fid, '%.6f\t%.6f\t%.4f\n', outputMatrix_cmp');
fclose(fid);

fprintf('对比点温度数据已导出至：%s\n', outputFile_cmp);



