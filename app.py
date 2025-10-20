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


# 模型路径配置 - 自动检测
def get_model_dir():
    """自动检测模型目录路径（优化部署兼容性）"""
    # 优先使用相对路径（适应Streamlit Cloud）
    possible_paths = [
        # 移除本地绝对路径，避免部署时干扰
        os.path.join(".", "saved_models_selected_features"),  # 当前工作目录下的子文件夹
        os.path.join(os.path.dirname(__file__), "saved_models_selected_features")  # 与app.py同级的文件夹
    ]

    for path in possible_paths:
        # 检查路径是否存在，且确保是目录
        if os.path.exists(path) and os.path.isdir(path):
            return path

    # 路径不存在时，返回默认相对路径并提示
    default_path = os.path.join(".", "saved_models_selected_features")
    st.warning(f"模型目录未找到，将尝试使用默认路径: {default_path}")
    return default_path


MODEL_DIR = get_model_dir()

# 特征标签映射
FEATURE_LABELS = {
    'age': {'label': '年龄', 'type': 'number', 'min': 45, 'max': 120},
    'gender': {'label': '性别', 'options': {'0': '女性', '1': '男性'}},
    'education': {'label': '教育水平', 'options': {'1': '低于初中', '2': '高中和职业', '3': '高等教育'}},
    'smoke': {'label': '吸烟', 'options': {'0': '否', '1': '是'}},
    'digeste': {'label': '胃病', 'options': {'0': '否', '1': '是'}},
    'lunge': {'label': '肺病', 'options': {'0': '否', '1': '是'}},
    'arthre': {'label': '关节炎', 'options': {'0': '否', '1': '是'}},
    'chronum': {'label': '慢性病数量', 'type': 'number', 'min': 0, 'max': 20},
    'adl': {'label': 'ADL评分', 'type': 'number', 'min': 0, 'max': 6, 'desc': '日常生活活动困难项数'},
    'iadl': {'label': 'IADL评分', 'type': 'number', 'min': 0, 'max': 6, 'desc': '工具性日常活动困难项数'},
    'cog': {'label': '认知功能评分', 'type': 'number', 'min': 0, 'max': 21, 'desc': '分数越高认知越好'},
    'cesd': {'label': 'CESD抑郁评分', 'type': 'number', 'min': 0, 'max': 30, 'desc': '分数越高抑郁越严重'},
    'selfhealth': {'label': '自评健康', 'options': {'1': '很差', '2': '差', '3': '一般', '4': '好', '5': '很好'}},
    'lonely': {'label': '孤独频率', 'options': {'1': '很少', '2': '有时', '3': '经常', '4': '总是'}},
    'lifesat': {'label': '生活满意度',
                'options': {'5': '极其满意', '4': '非常满意', '3': '比较满意', '2': '不太满意', '1': '一点也不满意'}},
    'hchild': {'label': '健在子女数', 'type': 'number', 'min': 0, 'max': 20}
}


@st.cache_resource
def load_models():
    """加载所有必需的模型和预处理器"""
    try:
        # 检查目录是否存在
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

        # 检查缺失特征
        missing_features = [f for f in selected_features if f not in data]
        if missing_features:
            raise ValueError(f"缺少必需的特征: {', '.join(missing_features)}")

        # 只提取重要特征
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

        # 确保列顺序一致
        X_processed = X_processed[selected_features]

        return X_processed

    except Exception as e:
        st.error(f"预处理错误: {str(e)}")
        raise


def generate_shap_plot(shap_values, feature_values, base_value, features_info):
    """生成SHAP瀑布图"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(10, 8))

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
            'chronum': '慢性病数量', 'smoke': '吸烟', 'digeste': '消化疾病',
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

    # 侧边栏 - 模型信息
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
                            f"{label} ({feature})",
                            options=options_display,
                            key=feature
                        )

                        # 找到对应的值
                        selected_idx = options_display.index(selected)
                        input_data[feature] = float(options_list[selected_idx])
                    else:
                        min_val = label_info.get('min', 0)
                        max_val = label_info.get('max', 100)
                        desc = label_info.get('desc', '')

                        help_text = desc if desc else None

                        input_data[feature] = st.number_input(
                            f"{label} ({feature})",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(min_val),
                            step=0.1,
                            help=help_text,
                            key=feature
                        )

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
                        st.caption("SHAP值显示每个特征对睡眠质量风险预测的贡献。红色表示增加风险，蓝色表示降低风险。")

                except Exception as e:
                    st.warning(f"特征影响分析生成失败: {str(e)}")

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
