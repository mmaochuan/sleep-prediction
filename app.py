import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import shap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# 页面配置
st.set_page_config(
    page_title="睡眠质量预测系统",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-color: #4caf50;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-color: #ff9800;
    }
    .risk-high {
        background-color: #ffebee;
        border-color: #f44336;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 3em;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #667eea;
        font-size: 1.5em;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# 模型路径配置
def get_model_dir():
    """自动检测模型目录路径（优化部署兼容性）"""
    possible_paths = [
        os.path.join(".", "saved_models_selected_features"),
        os.path.join(os.path.dirname(__file__), "saved_models_selected_features")
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path

    default_path = os.path.join(".", "saved_models_selected_features")
    st.warning(f"模型目录未找到，将尝试使用默认路径: {default_path}")
    return default_path


MODEL_DIR = get_model_dir()

# 特征标签映射（更新版）
FEATURE_LABELS = {
    'age': {
        'label': '年龄', 
        'type': 'number', 
        'min': 45, 
        'max': 120,
        'step': 1,
        'is_integer': True
    },
    'gender': {
        'label': '性别', 
        'options': {'0': '女性', '1': '男性'}
    },
    'education': {
        'label': '教育水平', 
        'options': {'1': '低于初中', '2': '高中和职业', '3': '高等教育'}
    },
    'smoke': {
        'label': '吸烟', 
        'options': {'0': '否', '1': '是'}
    },
    'digeste': {
        'label': '胃病', 
        'options': {'0': '否', '1': '是'}
    },
    'lunge': {
        'label': '肺病', 
        'options': {'0': '否', '1': '是'}
    },
    'arthre': {
        'label': '关节炎', 
        'options': {'0': '否', '1': '是'}
    },
    'chronum': {
        'label': '多病共存数量', 
        'type': 'number', 
        'min': 0, 
        'max': 14,
        'step': 1,
        'is_integer': True,
        'desc': '''**多病共存包括以下14种疾病：**
        
1. 高血压
2. 血脂异常
3. 糖尿病
4. 癌症
5. 肺病
6. 肝脏疾病
7. 心脏病
8. 中风
9. 肾脏疾病
10. 胃病
11. 精神疾病
12. 记忆疾病
13. 关节炎
14. 哮喘病

**请输入患有上述疾病的总数量（0-14）**'''
    },
    'adl': {
        'label': 'ADL评分', 
        'type': 'number', 
        'min': 0, 
        'max': 6,
        'step': 1,
        'is_integer': True,
        'desc': '''**ADL（日常生活活动能力）评分说明：**

ADL评估6项基本日常生活活动的困难程度：

1. **穿衣**：自己穿衣服有无困难
2. **洗澡**：自己洗澡有无困难
3. **进食**：自己吃饭有无困难
4. **转移**：上下床或椅子有无困难
5. **如厕**：自己上厕所有无困难
6. **控制大小便**：控制大小便有无困难

**计分方法：**
- 每项活动如果有困难，计1分
- 总分范围：0-6分
- 分数越高，表示日常生活能力越差

**请输入有困难的项目数量（0-6）**'''
    },
    'iadl': {
        'label': 'IADL评分', 
        'type': 'number', 
        'min': 0, 
        'max': 5,
        'step': 1,
        'is_integer': True,
        'desc': '''**IADL（工具性日常生活活动能力）评分说明：**

IADL评估5项工具性日常生活活动的困难程度：

1. **做家务**：做家务活有无困难
2. **做饭**：做饭有无困难
3. **购物**：购物有无困难
4. **管理钱财**：管理钱财有无困难
5. **吃药**：按时吃药有无困难

**计分方法：**
- 每项活动如果有困难，计1分
- 总分范围：0-5分
- 分数越高，表示工具性日常活动能力越差

**请输入有困难的项目数量（0-5）**'''
    },
    'cog': {
        'label': '认知功能评分', 
        'type': 'number', 
        'min': 0, 
        'max': 21,
        'step': 0.5,
        'is_integer': False,
        'desc': '''**CHARLS认知功能评分说明：**

认知功能总分由两部分组成，满分21分：

**一、精神状态（Mental Status，0-11分）**

1. **时间定向**（共3分）
   - 今天是几号？（年、月、日各1分）
   
2. **时间定向**（1分）
   - 今天是星期几？
   
3. **时间定向**（1分）
   - 现在是什么季节？（春夏秋冬）
   
4. **计算能力**（共5分）
   - 从100开始，连续减5次7
   - 即：100-7=93, 93-7=86, 86-7=79, 79-7=72, 72-7=65
   - 每答对一次得1分，最高5分
   
5. **视空间能力**（1分）
   - 临摹两个重叠的五边形
   - 能正确画出得1分

**二、情景记忆（Episodic Memory，0-10分）**

1. **立即回忆**（Immediate Recall，0-10分）
   - 访员读10个词（如：苹果、桌子、书等）
   - 受访者立即回忆
   - 每记对1个词得1分，最高10分
   
2. **延迟回忆**（Delayed Recall，0-10分）
   - 若干分钟后再次要求回忆同一组词
   - 每记对1个词得1分，最高10分
   
3. **最终得分计算**
   - 情景记忆得分 =（立即回忆得分 + 延迟回忆得分）÷ 2
   - 范围：0-10分
   - ⚠️ **注意：因为是平均值，所以可能出现小数（如5.5分）**

**总分计算：**
- 总分 = 精神状态得分（0-11分）+ 情景记忆得分（0-10分）
- 总分范围：0-21分
- **可以输入小数，如10.5、15.5等**
- **分数越高，认知功能越好**

**请输入总分（0-21，可含小数）**'''
    },
    'cesd': {
        'label': 'CESD抑郁评分', 
        'type': 'number', 
        'min': 0, 
        'max': 30,
        'step': 1,
        'is_integer': True,
        'desc': '''**CESD-10抑郁量表评分说明：**

包括10个问题，评估过去一周的感受：

**评分标准（每题1-4分）：**
- **1分** = 很少或根本没有（<1天）
- **2分** = 不太多（1-2天）
- **3分** = 有时或一半时间（3-4天）
- **4分** = 大多数时间（5-7天）

**10个问题：**

1. **DC009** 我因一些小事而烦恼
2. **DC010** 我在做事时很难集中精力
3. **DC011** 我感到情绪低落
4. **DC012** 我觉得做任何事都很费劲
5. **DC013** 我对未来充满希望 ⭐（反向题）
6. **DC014** 我感到害怕
7. **DC015** 我的睡眠不好
8. **DC016** 我很愉快 ⭐（反向题）
9. **DC017** 我感到孤独
10. **DC018** 我觉得我无法继续我的生活

**反向题计分（第5题、第8题）：**
- 原始1分 → 计为3分
- 原始2分 → 计为2分
- 原始3分 → 计为1分
- 原始4分 → 计为0分

**抑郁风险水平判定：**
- **0-9分**：无明显抑郁症状
- **10-12分**：轻度抑郁倾向
- **≥13分**：明显抑郁症状（可能存在抑郁障碍）

**总分范围：0-30分**
**分数越高，抑郁程度越严重**

**请输入总分（0-30）**'''
    },
    'selfhealth': {
        'label': '自评健康', 
        'options': {'1': '很差', '2': '差', '3': '一般', '4': '好', '5': '很好'}
    },
    'lonely': {
        'label': '孤独频率', 
        'options': {'1': '很少', '2': '有时', '3': '经常', '4': '总是'}
    },
    'lifesat': {
        'label': '生活满意度',
        'options': {'5': '极其满意', '4': '非常满意', '3': '比较满意', '2': '不太满意', '1': '一点也不满意'}
    },
    'hchild': {
        'label': '健在子女数', 
        'type': 'number', 
        'min': 0, 
        'max': 20,
        'step': 1,
        'is_integer': True
    }
}


@st.cache_resource
def load_models():
    """加载所有必需的模型和预处理器"""
    try:
        if not os.path.exists(MODEL_DIR):
            st.error(f"❌ 模型目录不存在: {MODEL_DIR}")
            return None, None, None, None, None

        # 加载特征信息
        selected_features_pkl = os.path.join(MODEL_DIR, 'selected_features.pkl')
        if os.path.exists(selected_features_pkl):
            selected_features_data = joblib.load(selected_features_pkl)
            features_info = {
                'selected_features': selected_features_data['selected_features'],
                'selected_categorical': selected_features_data.get('selected_categorical', []),
                'selected_continuous': selected_features_data.get('selected_continuous', [])
            }
        else:
            features_path = os.path.join(MODEL_DIR, 'model_features_info.json')
            with open(features_path, 'r', encoding='utf-8') as f:
                features_info = json.load(f)

        # 加载模型
        model_name = features_info.get('best_model_name')
        if not model_name:
            model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('best_model_') and f.endswith('.pkl')]
            if model_files:
                model_name = model_files[0].replace('best_model_', '').replace('.pkl', '')
            else:
                st.error("❌ 无法确定模型名称")
                return None, None, None, None, None

        model_path = os.path.join(MODEL_DIR, f'best_model_{model_name}.pkl')
        model = joblib.load(model_path)
        features_info['best_model_name'] = model_name

        # 加载编码器
        encoder_path = os.path.join(MODEL_DIR, 'ordinal_encoder.pkl')
        ordinal_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

        # 加载标准化器
        scaler_path = os.path.join(MODEL_DIR, 'scaler_continuous.pkl')
        scaler_cont = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # 初始化SHAP解释器
        explainer = shap.TreeExplainer(model)

        return model, ordinal_encoder, scaler_cont, features_info, explainer

    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None, None, None, None, None


def preprocess_input(data, features_info, ordinal_encoder, scaler_cont):
    """预处理输入数据"""
    try:
        selected_features = features_info['selected_features']
        selected_categorical = features_info.get('selected_categorical', [])
        selected_continuous = features_info.get('selected_continuous', [])

        missing_features = [f for f in selected_features if f not in data]
        if missing_features:
            raise ValueError(f"缺少必需的特征: {', '.join(missing_features)}")

        important_data = {k: v for k, v in data.items() if k in selected_features}
        df = pd.DataFrame([important_data])

        # 分类特征编码
        if selected_categorical and ordinal_encoder is not None:
            cat_encoded = pd.DataFrame(
                ordinal_encoder.transform(df[selected_categorical]),
                columns=selected_categorical
            )
        else:
            cat_encoded = pd.DataFrame()

        # 连续特征标准化
        if selected_continuous and scaler_cont is not None:
            cont_scaled = pd.DataFrame(
                scaler_cont.transform(df[selected_continuous]),
                columns=selected_continuous
            )
        else:
            cont_scaled = pd.DataFrame()

        # 合并特征
        if not cat_encoded.empty and not cont_scaled.empty:
            X_processed = pd.concat([cat_encoded, cont_scaled], axis=1)
        elif not cat_encoded.empty:
            X_processed = cat_encoded
        else:
            X_processed = cont_scaled

        X_processed = X_processed[selected_features]

        return X_processed

    except Exception as e:
        st.error(f"预处理错误: {str(e)}")
        raise


def configure_chinese_fonts():
    """配置中文字体显示"""
    import platform
    import matplotlib.font_manager as fm
    
    system = platform.system()
    
    # 获取系统可用字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # 定义不同平台的首选字体
    if system == 'Windows':
        preferred_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        preferred_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
    else:  # Linux / Cloud
        preferred_fonts = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
            'Droid Sans Fallback', 'AR PL UMing CN', 'Noto Sans SC'
        ]
    
    # 添加通用备选字体
    preferred_fonts.extend(['DejaVu Sans', 'sans-serif'])
    
    # 找到第一个可用的字体
    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        # 如果都不可用，使用所有备选字体
        plt.rcParams['font.sans-serif'] = preferred_fonts
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'


def generate_shap_plot(shap_values, feature_values, base_value, features_info):
    """生成SHAP瀑布图（改进中文字体支持）"""
    try:
        # 配置中文字体
        configure_chinese_fonts()

        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        feature_names = features_info['selected_features']
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]

        cumsum = base_value
        positions = []
        values = []
        colors = []
        labels = []

        feature_name_map = {
            'gender': '性别', 'age': '年龄', 'education': '教育程度',
            'cog': '认知功能', 'cesd': '抑郁评分', 'lonely': '孤独感',
            'selfhealth': '自评健康', 'depre': '抑郁程度', 'lifesat': '生活满意度',
            'chronum': '多病共存', 'smoke': '吸烟', 'digeste': '消化疾病',
            'lunge': '肺部疾病', 'arthre': '关节炎', 'hchild': '子女数量',
            'iadl': 'IADL评分', 'adl': 'ADL评分'
        }

        for idx in sorted_idx:
            positions.append(cumsum)
            values.append(shap_values[idx])
            colors.append('#FF6B6B' if shap_values[idx] > 0 else '#4ECDC4')

            feat_name = feature_name_map.get(feature_names[idx], feature_names[idx])
            feat_val = feature_values[idx]
            labels.append(f'{feat_name} = {feat_val:.2f}')

            cumsum += shap_values[idx]

        y_pos = np.arange(len(values))

        for i, (pos, val, color, label) in enumerate(zip(positions, values, colors, labels)):
            ax.barh(i, val, left=pos, color=color, alpha=0.8, height=0.6)
            text_x = pos + val / 2
            ax.text(text_x, i, f'{val:+.3f}',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        ax.axvline(base_value, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'基线值: {base_value:.3f}')
        ax.axvline(cumsum, color='red', linestyle='-', linewidth=2, alpha=0.7,
                   label=f'预测值: {cumsum:.3f}')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('SHAP值对预测的影响', fontsize=12, fontweight='bold')
        ax.set_title('特征对睡眠质量风险的影响分析', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"SHAP图生成失败: {str(e)}")
        return None


def main():
    """主函数"""

    # 加载模型
    model, ordinal_encoder, scaler_cont, features_info, explainer = load_models()

    if model is None:
        st.error("❌ 模型加载失败，请检查模型路径和文件")
        st.stop()

    # 标题
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 12px; color: white; margin-bottom: 2rem;'>
        <h1>🌙 睡眠质量预测系统</h1>
        <p style='font-size: 1.2em; margin-top: 1rem;'>基于机器学习的老年人睡眠质量风险评估</p>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 模型信息")
        st.info(f"""
        **模型类型**: {features_info['best_model_name']}  
        **特征数量**: {len(features_info['selected_features'])}  
        **AUC**: {features_info.get('best_auc', 'N/A')}
        """)

        st.markdown("### 📋 使用说明")
        st.write("""
        1. 填写所有必需的健康信息
        2. 点击"开始预测"按钮
        3. 查看风险评估结果
        4. 根据建议采取预防措施
        
        💡 **提示**：点击输入框旁的 ❓ 查看详细说明
        """)

    # 主要内容区域
    selected_features = features_info['selected_features']

    # 将特征分类
    categories = {
        '基本信息': ['gender', 'age', 'education'],
        '健康状况': ['smoke', 'digeste', 'lunge', 'arthre', 'chronum'],
        '功能评估': ['adl', 'iadl', 'cog', 'cesd'],
        '主观评价': ['selfhealth', 'lonely', 'lifesat'],
        '家庭信息': ['hchild']
    }

    # 创建表单
    with st.form("prediction_form"):
        input_data = {}

        for category, features in categories.items():
            important_features = [f for f in features if f in selected_features]

            if not important_features:
                continue

            st.markdown(f"<div class='section-header'>📋 {category}</div>", unsafe_allow_html=True)

            cols = st.columns(2)

            for idx, feature in enumerate(important_features):
                if feature not in FEATURE_LABELS:
                    continue

                label_info = FEATURE_LABELS[feature]
                label = label_info['label']

                with cols[idx % 2]:
                    if 'options' in label_info:
                        options_dict = label_info['options']
                        options_list = list(options_dict.keys())
                        options_display = [f"{options_dict[k]}" for k in options_list]

                        selected = st.selectbox(
                            f"{label}",
                            options=options_display,
                            key=feature
                        )

                        selected_idx = options_display.index(selected)
                        input_data[feature] = float(options_list[selected_idx])
                    else:
                        min_val = label_info.get('min', 0)
                        max_val = label_info.get('max', 100)
                        step = label_info.get('step', 1)
                        desc = label_info.get('desc', '')
                        is_integer = label_info.get('is_integer', False)

                        help_text = desc if desc else None

                        # 使用整数作为默认值和步长
                        value = st.number_input(
                            f"{label}",
                            min_value=int(min_val) if is_integer else float(min_val),
                            max_value=int(max_val) if is_integer else float(max_val),
                            value=int(min_val) if is_integer else float(min_val),
                            step=int(step) if is_integer else float(step),
                            help=help_text,
                            key=feature
                        )
                        
                        # 存储为float以保持一致性
                        input_data[feature] = float(value)

        # 提交按钮
        submitted = st.form_submit_button("🔮 开始预测", use_container_width=True)

    # 处理预测
    if submitted:
        with st.spinner('🔄 正在计算中...'):
            try:
                # 预处理
                X = preprocess_input(input_data, features_info, ordinal_encoder, scaler_cont)

                # 预测
                probability = model.predict_proba(X)[0, 1]
                risk_score = probability * 100

                # 风险分类
                if risk_score < 25:
                    risk_class = "低风险"
                    risk_color = "risk-low"
                    description = "该患者未来两年出现睡眠质量问题的风险较低。当前睡眠状况良好，建议继续保持健康的生活方式。"
                    recommendations = """
                    - 保持规律作息，每天固定时间睡觉和起床
                    - 坚持适度运动，如散步、太极拳等
                    - 保持均衡饮食，避免睡前摄入咖啡因
                    - 维持良好的心理状态，积极参与社交活动
                    - 定期体检，监测健康状况
                    """
                elif risk_score < 35:
                    risk_class = "中等风险"
                    risk_color = "risk-medium"
                    description = "该患者未来两年出现睡眠质量问题的风险中等。需要引起重视并采取预防措施，避免风险进一步升高。"
                    recommendations = """
                    - **建立良好的睡眠卫生习惯**：保持卧室环境舒适、安静、黑暗
                    - **控制慢性疾病**：定期就医，按医嘱服药
                    - **增加社交活动**：参与社区活动，减少孤独感
                    - **心理健康关注**：如有抑郁、焦虑症状，及时咨询心理医生
                    - **避免不良习惯**：戒烟限酒，规律作息
                    - **定期随访**：每3-6个月复查一次
                    """
                else:
                    risk_class = "高风险"
                    risk_color = "risk-high"
                    description = "该患者未来两年出现睡眠质量问题的风险较高。强烈建议立即采取干预措施并密切监测睡眠状况。"
                    recommendations = """
                    - **及时就医**：建议到医院睡眠科进行专业评估
                    - **积极治疗基础疾病**：控制高血压、糖尿病等慢性病
                    - **心理干预**：必要时接受心理咨询或认知行为疗法
                    - **药物治疗**：在医生指导下使用助眠药物
                    - **生活方式调整**：严格作息时间，避免白天长时间午睡
                    - **社会支持**：寻求家人、朋友的情感支持
                    - **密切随访**：每月复查，及时调整治疗方案
                    """

                # 显示结果
                st.markdown("---")
                st.markdown("## 📊 预测结果")

                # 风险评分展示
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class='metric-container'>
                        <div>睡眠质量风险评分</div>
                        <div class='metric-value'>{risk_score:.1f}</div>
                        <div style='font-size: 1.2em;'>{risk_class}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 风险说明
                st.markdown(f"""
                <div class='risk-box {risk_color}'>
                    <h3>🎯 风险等级: {risk_class}</h3>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)

                # 建议
                st.markdown("### 💡 健康建议")
                st.info(recommendations)

                # SHAP解释
                st.markdown("### 📈 特征影响分析")

                try:
                    shap_values = explainer.shap_values(X)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]

                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

                    fig = generate_shap_plot(shap_values[0], X.values[0], base_value, features_info)

                    if fig:
                        st.pyplot(fig)
                        st.caption("📌 SHAP值显示每个特征对睡眠质量风险预测的贡献。红色表示增加风险，蓝色表示降低风险。")

                except Exception as e:
                    st.warning(f"⚠️ 特征影响分析生成失败: {str(e)}")

                # 预测详情
                with st.expander("📋 查看预测详情"):
                    st.write("**输入数据:**")
                    st.json(input_data)
                    st.write(f"**预测概率:** {probability:.4f}")
                    st.write(f"**预测时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            except Exception as e:
                st.error(f"❌ 预测失败: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


if __name__ == '__main__':
    main()
