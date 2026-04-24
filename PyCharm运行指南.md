# PyCharm 运行说明

## 快速启动

### 方式一：直接运行

1. 打开 PyCharm，`File` → `Open`，选择项目目录
2. 在项目树中右键 `app.py`，选择 `Run 'app'`
3. 或点击右上角绿色运行按钮 ▶️

### 方式二：配置运行（推荐）

1. 右上角运行配置下拉 → `Edit Configurations...`
2. 点击 `+`，选择 `Python`
3. 配置：
   - Name: Flask Visualization Platform
   - Script path: 选择 `app.py`
   - Working directory: 选择项目根目录
   - Python interpreter: 已安装依赖的解释器
4. 保存后点击运行

### 方式三：终端

在 PyCharm 终端（Alt + F12）中执行：

```bash
python app.py
```

## 依赖检查

1. `File` → `Settings` → `Project` → `Python Interpreter`
2. 确认已安装：Flask, flask-cors, numpy, pandas
3. 若未安装：`pip install -r requirements.txt`

## 运行成功标志

```
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

## 访问

- http://localhost:5000
- http://127.0.0.1:5000

## 常见问题

- **ModuleNotFoundError**：安装依赖
- **端口占用**：修改 `app.py` 中 `port=5001`
- **模板找不到**：确认 templates 与 static 在项目根目录
