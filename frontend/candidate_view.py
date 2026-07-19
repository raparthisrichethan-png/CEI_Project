import streamlit as st
import os
import plotly.express as px
from backend.parser import extract_text, segment_resume, extract_contact_info
from backend.similarity_engine import SimilarityEngine
from frontend.ui_components import render_score_radial, render_glass_card, render_bullet_list

def calculate_ats_score(parsed_resume, contact_info):
    """
    Calculate an ATS Score out of 100 based on resume formatting and structure:
    - Standard section detection (up to 40 pts)
    - Contact details completeness (up to 30 pts)
    - Length and email validation (up to 30 pts)
    """
    score = 0
    reasons = []
    
    # 1. Structure Detection
    sections = ['skills', 'experience', 'education', 'projects']
    sections_found = 0
    for sec in sections:
        if parsed_resume.get(sec, "").strip():
            sections_found += 1
            
    score += sections_found * 10
    reasons.append(f"Found {sections_found}/4 core standard resume sections (Skills, Experience, Education, Projects). (+{sections_found*10} pts)")
    
    # 2. Contact details
    if contact_info.get("email"):
        score += 10
        reasons.append("Email address detected. (+10 pts)")
    else:
        reasons.append("Missing email address. (-10 pts)")
        
    if contact_info.get("phone"):
        score += 10
        reasons.append("Phone number detected. (+10 pts)")
    else:
        reasons.append("Missing contact phone number. (-10 pts)")
        
    if contact_info.get("linkedin") or contact_info.get("github"):
        score += 10
        reasons.append("Social links (LinkedIn/GitHub) detected. (+10 pts)")
    else:
        reasons.append("No portfolio or LinkedIn links detected. (-10 pts)")
        
    # 3. Format/Length Heuristics
    # Check if skills list is reasonably detailed
    skills_len = len(parsed_resume.get("skills", "").split(","))
    if skills_len >= 5:
        score += 15
        reasons.append(f"Extracted list of {skills_len} distinct skills. (+15 pts)")
    elif skills_len > 0:
        score += 5
        reasons.append("Skills section contains too few keywords. (+5 pts)")
    else:
        reasons.append("No explicit skills section found. (0 pts)")
        
    # Check experience length
    exp_len = len(parsed_resume.get("experience", "").split())
    if exp_len > 100:
        score += 15
        reasons.append("Detailed experience bullet points detected. (+15 pts)")
    elif exp_len > 10:
        score += 5
        reasons.append("Experience section is very brief. (+5 pts)")
    else:
        reasons.append("No work experience details found. (0 pts)")

    return min(100, score), reasons

def run_candidate_view(rag_pipeline):
    st.header("Candidate Dashboard")
    st.markdown("Upload your resume and the target Job Description to analyze your match, get personalized roadmap recommendations, and chat with your hiring assistant.")

    # 1. Inputs
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Upload Resume")
        uploaded_resume = st.file_uploader("Upload your Resume (.pdf, .docx)", type=["pdf", "docx"], key="cand_resume")
    with col2:
        st.subheader("2. Target Job Description")
        uploaded_jd = st.file_uploader("Upload Target Job Description File (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], key="cand_jd_uploader")
        st.markdown("**OR Enter Text Directly:**")
        jd_text_direct = st.text_area("Target JD Text", height=100, key="cand_jd_text", placeholder="Paste target Job Description here...")

    # Determine JD Text
    jd_text = ""
    if uploaded_jd:
        temp_jd_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", f"temp_cand_jd_{uploaded_jd.name}")
        with open(temp_jd_path, "wb") as f:
            f.write(uploaded_jd.getbuffer())
        jd_text = extract_text(temp_jd_path)
        try:
            os.remove(temp_jd_path)
        except Exception:
            pass
    elif jd_text_direct.strip():
        jd_text = jd_text_direct.strip()

    if not uploaded_resume or not jd_text:
        st.info("Please upload both a resume and provide a target job description to begin.")
        return

    # 2. Parsing and Similarity computation
    with st.spinner("Processing documents locally..."):
        # Save resume temporarily
        temp_res_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "resumes")
        os.makedirs(temp_res_dir, exist_ok=True)
        temp_res_path = os.path.join(temp_res_dir, f"temp_cand_{uploaded_resume.name}")
        with open(temp_res_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())

        raw_resume_text = extract_text(temp_res_path)
        parsed_resume = segment_resume(raw_resume_text, api_key=rag_pipeline.api_key)
        contact_info = extract_contact_info(raw_resume_text, api_key=rag_pipeline.api_key)

        # Cleanup temp resume
        try:
            os.remove(temp_res_path)
        except Exception:
            pass

        # Parse JD via LLM (cached)
        parsed_jd = rag_pipeline.extract_jd_requirements(jd_text)

        # Compute semantic match scores
        similarity_engine = SimilarityEngine()
        metrics = similarity_engine.calculate_scores(parsed_resume, parsed_jd)
        
        # Calculate ATS score
        ats_score, ats_reasons = calculate_ats_score(parsed_resume, contact_info)

    # 3. Main Scoring dashboard
    tab_dash, tab_sections, tab_roadmap, tab_chat = st.tabs([
        "📊 Score Dashboard", 
        "🔍 Section Analysis", 
        "📈 Learning Roadmap", 
        "💬 AI Hiring Chat"
    ])

    with tab_dash:
        st.subheader("Your Matching Score Summary")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            render_score_radial(metrics["overall_score"], "Overall Resume Match")
        with col_g2:
            render_score_radial(ats_score, "ATS Compatibility Score")

        st.markdown("### Profile Segment Match breakdown")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"**Skills Match:** {metrics['skills_match']:.1f}%")
            st.progress(metrics['skills_match'] / 100.0)
            
            st.markdown(f"**Experience Match:** {metrics['experience_match']:.1f}%")
            st.progress(metrics['experience_match'] / 100.0)

            st.markdown(f"**Education Match:** {metrics['education_match']:.1f}%")
            st.progress(metrics['education_match'] / 100.0)
        
        with col_p2:
            st.markdown(f"**Projects Match:** {metrics['projects_match']:.1f}%")
            st.progress(metrics['projects_match'] / 100.0)

            st.markdown(f"**Certifications Match:** {metrics['certifications_match']:.1f}%")
            st.progress(metrics['certifications_match'] / 100.0)

        # ATS Checklist
        st.markdown("### ATS Structure Checklist")
        render_bullet_list("ATS Optimization Details", ats_reasons, icon="✔", icon_color="#10B981")

    with tab_sections:
        st.subheader("Segment Analysis & Extracted Content")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Your Resume Content (Extracted)")
            sec_sel = st.selectbox("Select Resume Section to View", ["Skills", "Experience", "Education", "Projects", "Certifications", "Contact Info"])
            if sec_sel == "Skills":
                st.info(parsed_resume.get("skills") or "No skills section found.")
            elif sec_sel == "Experience":
                st.info(parsed_resume.get("experience") or "No experience section found.")
            elif sec_sel == "Education":
                st.info(parsed_resume.get("education") or "No education section found.")
            elif sec_sel == "Projects":
                st.info(parsed_resume.get("projects") or "No projects section found.")
            elif sec_sel == "Certifications":
                st.info(parsed_resume.get("certifications") or "No certifications section found.")
            elif sec_sel == "Contact Info":
                st.json(contact_info)
        
        with col_s2:
            st.markdown("#### Job Description Requirements (Extracted)")
            jd_sec_sel = st.selectbox("Select JD Requirement to View", ["Skills", "Experience", "Education", "Projects", "Certifications"])
            key_map = {
                "Skills": "skills",
                "Experience": "experience",
                "Education": "education",
                "Projects": "projects",
                "Certifications": "certifications"
            }
            st.success(parsed_jd.get(key_map[jd_sec_sel]) or "Requirement not specified.")

        st.markdown("---")
        st.subheader("AI Match Analysis Report")
        
        if rag_pipeline.has_api_key():
            if st.button("Generate Detailed AI Screening Feedback", key="cand_gen_feedback"):
                with st.spinner("Analyzing resume against JD..."):
                    feedback = rag_pipeline.generate_analysis(parsed_resume, parsed_jd)
                    st.markdown(feedback)
        else:
            st.info("Provide a Gemini API Key in the sidebar to generate detailed strengths, weaknesses, and rewrite suggestions.")

    with tab_roadmap:
        st.subheader("Personalized Learning Roadmap & Recommendations")
        st.markdown("Bridge your skill gaps with a customized plan:")
        
        if rag_pipeline.has_api_key():
            if st.button("Generate My Roadmap & Project Ideas", key="cand_gen_roadmap"):
                with st.spinner("Designing your roadmap..."):
                    roadmap = rag_pipeline.generate_roadmap(parsed_resume, parsed_jd)
                    st.markdown(roadmap)
        else:
            st.info("Provide a Gemini API Key in the sidebar to generate customized learning roadmaps, project recommendations, and certificates.")

    with tab_chat:
        st.subheader("Chat with AI Hiring Assistant")
        st.markdown("Ask questions about your resume match, how to improve, what skills you need, or ask for resume bullet-point suggestions.")

        # Initialize Chat History
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": f"Hi! I've analyzed your resume against the target role. You have an overall match score of {metrics['overall_score']:.1f}% and an ATS score of {ats_score}/100. How can I help you improve today?"}
            ]

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat Input
        if user_input := st.chat_input("Ask a question (e.g. 'How can I improve my experience section?'):"):
            with st.chat_message("user"):
                st.write(user_input)
            
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Query pipeline
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_pipeline.run_chatbot(
                        parsed_resume=parsed_resume,
                        parsed_jd=parsed_jd,
                        metrics=metrics,
                        chat_history=st.session_state.chat_history,
                        user_question=user_input
                    )
                    st.write(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
