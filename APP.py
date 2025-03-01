# 导入需要的库
import streamlit as st
import pandas as pd
import joblib

# Streamlit 用户界面
st.markdown("""
<style>
    .custom-title {
        font-size: 16px; 
        font-family: 'Arial', sans-serif; 
    }
</style>
""", unsafe_allow_html=True)
 
st.markdown("""
<div class="custom-title">
    基于机器学习的帕金森病检测系统
</div>
""", unsafe_allow_html=True)
 
# 创建输入框
st.markdown("""
<div class="custom-title">
    请输入以下特征值：
</div>
""", unsafe_allow_html=True)
a = st.number_input("PPE", step=0.01,value=0.15)
b = st.number_input("spread1",step=0.01,value=-6.18)
c = st.number_input("MDVP:Fo(Hz)",step=0.01,value=171.04)
d = st.number_input("spread2", step=0.01,value=0.23)
e = st.number_input("MDVP:Fhi(Hz)", step=0.01,value=208.31)
f = st.number_input("MDVP:APQ",step=0.01,value=0.02)
g = st.number_input("Jitter:DDP", step=0.01,value=0.01)
h = st.number_input("MDVP:Jitter(Abs)",step=0.01,value=0.00)
# 如果按下按钮
if st.button("预测"):  # 显示按钮
    # 加载训练好的模型
    model = joblib.load("XGBoost.pkl")
    # 将输入存储DataFrame
    X = pd.DataFrame([[a,b,c,d,e,f,g,h]],
                     columns = ['PPE', 'spread1', 'MDVP:Fo(Hz)', 'spread2', 'MDVP:Fhi(Hz)', 'MDVP:APQ',
       'Jitter:DDP', 'MDVP:Jitter(Abs)'])
    # 进行预测
    prediction = model.predict(X)[0]
    Predict_proba = model.predict_proba(X)[:, 1][0]
    # 输出预测结果
    if prediction == 0:
        st.subheader(f"帕金森患者:  否")
    else:
        st.subheader(f"帕金森患者:  是")
    st.subheader(f"帕金森患者概率:  {'%.2f' % float(Predict_proba * 100) + '%'}")

