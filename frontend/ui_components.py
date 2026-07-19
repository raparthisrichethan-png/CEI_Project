import streamlit as st
import plotly.graph_objects as go

def render_score_radial(score, label):
    """Render a premium radial/gauge chart using Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': label, 'font': {'size': 20, 'color': '#1F2937', 'family': 'Outfit, Inter, sans-serif'}},
        number={'suffix': "%", 'font': {'size': 44, 'color': '#0F172A', 'family': 'Outfit, Inter, sans-serif', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#3B82F6", 'thickness': 0.25},
            'bgcolor': "#E2E8F0",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#FEE2E2'},
                {'range': [50, 75], 'color': '#FEF3C7'},
                {'range': [75, 100], 'color': '#D1FAE5'}
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
    )
    st.plotly_chart(fig, use_container_width=True)

def clean_html(html_str):
    """Strip all newlines and leading spaces to prevent markdown code block formatting."""
    return "".join(line.strip() for line in html_str.split("\n"))

def render_glass_card(title, value, subtext="", status="info"):
    """Render a premium, modern HTML card in Streamlit."""
    colors = {
        "success": {
            "bg": "linear-gradient(135deg, rgba(209, 250, 229, 0.4) 0%, rgba(167, 243, 208, 0.4) 100%)",
            "border": "#10B981",
            "text": "#065F46"
        },
        "warning": {
            "bg": "linear-gradient(135deg, rgba(254, 243, 199, 0.4) 0%, rgba(253, 230, 138, 0.4) 100%)",
            "border": "#F59E0B",
            "text": "#92400E"
        },
        "danger": {
            "bg": "linear-gradient(135deg, rgba(254, 226, 226, 0.4) 0%, rgba(254, 202, 202, 0.4) 100%)",
            "border": "#EF4444",
            "text": "#991B1B"
        },
        "info": {
            "bg": "linear-gradient(135deg, rgba(219, 234, 254, 0.4) 0%, rgba(191, 219, 254, 0.4) 100%)",
            "border": "#3B82F6",
            "text": "#1E40AF"
        }
    }
    
    style = colors.get(status, colors["info"])
    subtext_html = f'<div style="font-size: 0.8rem; color: #64748B; margin-top: 5px; font-family: \'Inter\', sans-serif;">{subtext}</div>' if subtext else ''
    
    html = f"""
    <div style="
        background: {style['bg']};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {style['border']};
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    ">
        <h4 style="margin: 0; color: #475569; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'Inter', sans-serif;">{title}</h4>
        <div style="font-size: 2.25rem; font-weight: 800; color: {style['text']}; margin: 5px 0 0 0; font-family: 'Outfit', sans-serif;">{value}</div>
        {subtext_html}
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)

def render_bullet_list(title, items, icon="⚡", icon_color="#3B82F6"):
    """Render a beautiful custom bullet list."""
    list_items = ""
    for item in items:
        list_items += f"""
        <li style="margin-bottom: 10px; display: flex; align-items: flex-start; font-family: 'Inter', sans-serif; color: #334155; font-size: 0.95rem; line-height: 1.5;">
            <span style="color: {icon_color}; margin-right: 10px; font-weight: bold; flex-shrink: 0;">{icon}</span>
            <span>{item}</span>
        </li>
        """
    
    html = f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <h3 style="margin-top: 0; margin-bottom: 15px; color: #1E293B; font-size: 1.15rem; font-weight: 700; font-family: 'Outfit', sans-serif; border-bottom: 2px solid #F1F5F9; padding-bottom: 10px;">{title}</h3>
        <ul style="list-style: none; padding-left: 0; margin: 0;">
            {list_items}
        </ul>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)

def render_comparison_matrix(candidates, parsed_jd):
    """
    Render an attractive HTML comparison table.
    candidates: list of dicts with keys: 'name', 'skills_match', 'experience_match', 'education_match', 'overall_score', etc.
    """
    rows = ""
    for c in candidates:
        rows += f"""
        <tr style="border-bottom: 1px solid #E2E8F0; background-color: #FFFFFF; transition: background-color 0.2s;">
            <td style="padding: 15px; font-weight: 700; color: #0F172A; font-family: 'Outfit', sans-serif;">{c['name']}</td>
            <td style="padding: 15px; font-weight: bold; text-align: center; color: {get_score_color(c['overall_score'])}; font-family: 'Inter', sans-serif;">{c['overall_score']:.1f}%</td>
            <td style="padding: 15px; text-align: center; color: #334155; font-family: 'Inter', sans-serif;">{c['skills_match']:.1f}%</td>
            <td style="padding: 15px; text-align: center; color: #334155; font-family: 'Inter', sans-serif;">{c['experience_match']:.1f}%</td>
            <td style="padding: 15px; text-align: center; color: #334155; font-family: 'Inter', sans-serif;">{c['education_match']:.1f}%</td>
            <td style="padding: 15px; text-align: center; color: #334155; font-family: 'Inter', sans-serif;">{c['projects_match']:.1f}%</td>
            <td style="padding: 15px; text-align: center; color: #334155; font-family: 'Inter', sans-serif;">{c['certifications_match']:.1f}%</td>
        </tr>
        """
        
    html = f"""
    <div style="overflow-x: auto; border: 1px solid #E2E8F0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
        <table style="width: 100%; border-collapse: collapse; text-align: left; background: white; font-family: 'Inter', sans-serif; font-size: 0.9rem;">
            <thead>
                <tr style="background: #F8FAFC; border-bottom: 2px solid #E2E8F0; color: #475569;">
                    <th style="padding: 15px; font-weight: 600;">Candidate Name</th>
                    <th style="padding: 15px; font-weight: 600; text-align: center;">Overall Match</th>
                    <th style="padding: 15px; font-weight: 600; text-align: center;">Skills Match</th>
                    <th style="padding: 15px; font-weight: 600; text-align: center;">Experience Match</th>
                    <th style="padding: 15px; font-weight: 600; text-align: center;">Education Match</th>
                    <th style="padding: 15px; font-weight: 600; text-align: center;">Project Match</th>
                    <th style="padding: 15px; font-weight: 600; text-align: center;">Cert Match</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)

def get_score_color(score):
    if score >= 75:
        return "#10B981" # Emerald Green
    elif score >= 50:
        return "#F59E0B" # Amber Yellow
    else:
        return "#EF4444" # Red

def render_trust_index(eval_metrics):
    """
    Render a premium quality and alignment metrics index card (LLM-as-a-Judge RAG Validation).
    """
    if not eval_metrics:
        return
        
    f_score = eval_metrics.get("faithfulness_score", 100)
    f_reason = eval_metrics.get("faithfulness_reason", "Verified grounded in retrieved context.")
    r_score = eval_metrics.get("relevance_score", 100)
    r_reason = eval_metrics.get("relevance_reason", "Response addresses the query prompt accurately.")
    
    # Heuristic styling colors based on validation levels
    f_color = "#10b981" if f_score >= 80 else ("#f59e0b" if f_score >= 50 else "#ef4444")
    r_color = "#10b981" if r_score >= 80 else ("#f59e0b" if r_score >= 50 else "#ef4444")
    
    html = f"""
    <div style="
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.02);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">
            <span style="font-size: 1.2rem; margin-right: 8px;">🛡️</span>
            <span style="font-weight: 700; color: #1e293b; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em;">AI Trust & Alignment Index (LLM-as-a-Judge)</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <!-- Faithfulness Metric -->
            <div style="background: white; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; font-weight: 600; color: #475569; font-family: 'Inter', sans-serif;">Faithfulness (Groundedness)</span>
                    <span style="font-size: 0.9rem; font-weight: 700; color: {f_color}; font-family: 'Inter', sans-serif;">{f_score}%</span>
                </div>
                <div style="font-size: 0.78rem; color: #64748b; line-height: 1.4; font-family: 'Inter', sans-serif;">{f_reason}</div>
            </div>
            <!-- Relevance Metric -->
            <div style="background: white; border-radius: 8px; padding: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; font-weight: 600; color: #475569; font-family: 'Inter', sans-serif;">Answer Relevance</span>
                    <span style="font-size: 0.9rem; font-weight: 700; color: {r_color}; font-family: 'Inter', sans-serif;">{r_score}%</span>
                </div>
                <div style="font-size: 0.78rem; color: #64748b; line-height: 1.4; font-family: 'Inter', sans-serif;">{r_reason}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)
