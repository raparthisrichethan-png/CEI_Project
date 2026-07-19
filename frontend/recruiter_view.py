import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from backend.parser import extract_text, segment_resume, extract_contact_info, classify_document_type
from backend.similarity_engine import SimilarityEngine
from backend.config import DEFAULT_WEIGHTS, DEFAULT_LLM_MODEL
from frontend.ui_components import render_comparison_matrix, render_glass_card

@st.cache_data
def load_preloaded_dataset():
    """Load preloaded mock resumes (supports JSON and JSONL formats)."""
    dataset_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "resume_dataset.json")
    dataset_jsonl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "resumes_dataset.jsonl")
    
    candidates = []
    
    # Try loading the new large JSONL file first
    if os.path.exists(dataset_jsonl_path):
        try:
            with open(dataset_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        candidates.append({
                            "id": f"preloaded_{item.get('ResumeID', 'unknown')}",
                            "name": item.get("Name", "Unknown Candidate"),
                            "email": item.get("Email", "contact@email.com"),
                            "phone": item.get("Phone", ""),
                            "category": item.get("Category", "General"),
                            "source": "Pre-loaded Pool",
                            "parsed_resume": {
                                "skills": item.get("Skills", ""),
                                "experience": item.get("Experience", ""),
                                "education": item.get("Education", ""),
                                "projects": "",
                                "certifications": "",
                                "text": item.get("Text", ""),
                                "skills_segmented": False
                            }
                        })
            return candidates
        except Exception as e:
            st.error(f"Error loading large JSONL dataset: {e}")
            
    # Fallback to the small JSON file if JSONL doesn't exist
    if os.path.exists(dataset_json_path):
        try:
            with open(dataset_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    candidates.append({
                        "id": f"preloaded_{idx}",
                        "name": item.get("name", "Unknown Candidate"),
                        "email": item.get("email", ""),
                        "phone": item.get("phone", ""),
                        "category": item.get("category", "General"),
                        "source": "Pre-loaded Pool",
                        "parsed_resume": {
                            "skills": item.get("skills", ""),
                            "experience": item.get("experience", ""),
                            "education": item.get("education", ""),
                            "projects": item.get("projects", ""),
                            "certifications": item.get("certifications", ""),
                            "text": item.get("raw_text", ""),
                            "skills_segmented": True  # Already segmented
                        }
                    })
            return candidates
        except Exception as e:
            st.error(f"Error loading JSON dataset: {e}")
            
    return []

def run_recruiter_view(rag_pipeline):
    st.header("Recruiter Dashboard")
    st.markdown("Upload a Job Description, parse candidate resumes, adjust match criteria, and search the talent pool.")

    # 1. Job Description Inputs
    st.subheader("1. Job Description Setup")
    jd_input_method = st.radio("Input Job Description via:", ["Text Input", "Upload JD File (.pdf, .docx, .txt)"], horizontal=True)
    
    jd_text = ""
    if jd_input_method == "Text Input":
        jd_text = st.text_area(
            "Enter Job Description Requirements here:",
            height=200,
            placeholder="We are looking for a Senior Software Engineer with 5+ years of experience in Python, Django, AWS, Kubernetes, and PostgreSQL. A Bachelor's degree in Computer Science is required. Certifications like AWS Solutions Architect are a plus."
        )
    else:
        uploaded_jd = st.file_uploader("Upload Job Description File", type=["pdf", "docx", "txt"], key="jd_uploader")
        if uploaded_jd:
            # Save temporarily
            temp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", f"temp_jd_{uploaded_jd.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_jd.getbuffer())
            jd_text = extract_text(temp_path)
            try:
                os.remove(temp_path)
            except Exception:
                pass
            
            jd_type = classify_document_type(jd_text)
            if jd_type == 'resume':
                st.warning(f"⚠️ **Potential Misplacement:** The file uploaded in the Job Description slot ({uploaded_jd.name}) looks like a Resume. Please double check your upload.")
            else:
                st.success(f"Job Description extracted successfully ({len(jd_text)} characters).")

    if not jd_text.strip():
        st.warning("Please provide a Job Description to begin candidate evaluation.")
        return

    # Extract JD requirements once (cached using session state and hashing)
    import hashlib
    jd_hash = hashlib.md5(jd_text.encode('utf-8')).hexdigest()
    
    if "jd_hash" not in st.session_state or st.session_state.jd_hash != jd_hash:
        with st.spinner("Analyzing and structuring Job Description..."):
            extracted_jd = rag_pipeline.extract_jd_requirements(jd_text)
        st.session_state.jd_hash = jd_hash
        st.session_state.extracted_jd = extracted_jd
        # Initialize editable session state values
        for key in ["skills", "experience", "education", "projects", "certifications"]:
            st.session_state[f"recruiter_jd_{key}"] = extracted_jd.get(key, "")

    # Display extracted requirements in an expander with manual override fields
    with st.expander("🔍 Verify & Edit JD Requirements (Manual Override)", expanded=True):
        st.markdown("Review and adjust the extracted requirements below. If AI extraction failed, you can manually paste/type your requirements:")
        
        edited_skills = st.text_area("Skills Required:", value=st.session_state.get("recruiter_jd_skills", ""), key="recruiter_skills_area")
        edited_experience = st.text_area("Experience Required:", value=st.session_state.get("recruiter_jd_experience", ""), key="recruiter_experience_area")
        edited_education = st.text_area("Education Required:", value=st.session_state.get("recruiter_jd_education", ""), key="recruiter_education_area")
        edited_projects = st.text_area("Projects Required:", value=st.session_state.get("recruiter_jd_projects", ""), key="recruiter_projects_area")
        edited_certifications = st.text_area("Certifications Required:", value=st.session_state.get("recruiter_jd_certifications", ""), key="recruiter_certifications_area")
        
        # Save updates back to session state
        st.session_state["recruiter_jd_skills"] = edited_skills
        st.session_state["recruiter_jd_experience"] = edited_experience
        st.session_state["recruiter_jd_education"] = edited_education
        st.session_state["recruiter_jd_projects"] = edited_projects
        st.session_state["recruiter_jd_certifications"] = edited_certifications

    # Construct the final parsed_jd to be used downstream
    parsed_jd = {
        "skills": st.session_state["recruiter_jd_skills"],
        "experience": st.session_state["recruiter_jd_experience"],
        "education": st.session_state["recruiter_jd_education"],
        "projects": st.session_state["recruiter_jd_projects"],
        "certifications": st.session_state["recruiter_jd_certifications"]
    }

    st.markdown("---")

    # 2. Match Weight Configuration (Sidebar or Column)
    st.subheader("2. Configure Evaluation Weights")
    st.markdown("Adjust the importance of each resume section for the overall match calculation:")
    
    col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
    with col_w1:
        w_skills = st.slider("Skills Weight", 0, 100, int(DEFAULT_WEIGHTS["skills"] * 100), step=5) / 100.0
    with col_w2:
        w_experience = st.slider("Experience Weight", 0, 100, int(DEFAULT_WEIGHTS["experience"] * 100), step=5) / 100.0
    with col_w3:
        w_education = st.slider("Education Weight", 0, 100, int(DEFAULT_WEIGHTS["education"] * 100), step=5) / 100.0
    with col_w4:
        w_projects = st.slider("Projects Weight", 0, 100, int(DEFAULT_WEIGHTS["projects"] * 100), step=5) / 100.0
    with col_w5:
        w_certifications = st.slider("Certifications Weight", 0, 100, int(DEFAULT_WEIGHTS["certifications"] * 100), step=5) / 100.0

    # Normalize weights
    total_w = w_skills + w_experience + w_education + w_projects + w_certifications
    if total_w == 0:
        st.error("Weights cannot all be zero. Resetting to defaults.")
        custom_weights = DEFAULT_WEIGHTS
    else:
        custom_weights = {
            "skills": w_skills / total_w,
            "experience": w_experience / total_w,
            "education": w_education / total_w,
            "projects": w_projects / total_w,
            "certifications": w_certifications / total_w
        }

    st.markdown("---")

    # 3. Resume Uploads & Talent Pool Options
    st.subheader("3. Upload Candidates & Search Talent Pool")
    
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded_resumes = st.file_uploader(
            "Upload Candidate Resumes (Multiple files allowed)", 
            type=["pdf", "docx"], 
            accept_multiple_files=True,
            key="resumes_uploader"
        )
    with col_u2:
        include_preloaded = st.checkbox("Include Pre-loaded Talent Pool (Kaggle Resumes)", value=True, help="Includes sample resumes from our pre-populated database in the ranking list.")

    # Process all resumes
    candidates_pool = []
    
    # Add preloaded dataset if selected
    if include_preloaded:
        preloaded_data = load_preloaded_dataset()
        
        # Get list of unique categories
        categories = sorted(list(set(c["category"] for c in preloaded_data)))
        
        # UI controls for filtering the large talent pool
        st.markdown("##### Filter Talent Pool")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_category = st.selectbox("Job Category Filter:", ["All"] + categories)
        with col_f2:
            max_pool_size = st.slider("Max Candidates to Evaluate:", 5, 200, 30, step=5, help="Limit pool size for faster real-time similarity computations.")
        
        filtered_preloaded = preloaded_data
        if selected_category != "All":
            filtered_preloaded = [c for c in filtered_preloaded if c["category"] == selected_category]
            
        filtered_preloaded = filtered_preloaded[:max_pool_size]
        
        # Dynamically segment flat text if not already done
        for c in filtered_preloaded:
            text_to_segment = c["parsed_resume"].get("text", "")
            if text_to_segment and not c["parsed_resume"].get("skills_segmented", False):
                parsed_res = segment_resume(text_to_segment, api_key=rag_pipeline.api_key)
                c["parsed_resume"] = parsed_res
                c["parsed_resume"]["skills_segmented"] = True
                c["parsed_resume"]["text"] = text_to_segment
        
        candidates_pool.extend(filtered_preloaded)

    # Save and parse uploaded resumes
    if uploaded_resumes:
        for idx, file in enumerate(uploaded_resumes):
            # Write to temp file
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "resumes")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"uploaded_{file.name}")
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())

            # Extract raw text & parse
            raw_text = extract_text(temp_path)
            res_type = classify_document_type(raw_text)
            if res_type == 'jd':
                st.warning(f"⚠️ **Potential Misplacement:** The candidate file ({file.name}) looks like a Job Description rather than a Resume. Please review the upload.")
            
            parsed_resume = segment_resume(raw_text, api_key=rag_pipeline.api_key)
            contact = extract_contact_info(raw_text, api_key=rag_pipeline.api_key)

            # Cleanup
            try:
                os.remove(temp_path)
            except Exception:
                pass

            candidates_pool.append({
                "id": f"uploaded_{idx}_{file.name}",
                "name": contact["name"] if contact["name"] != "Unknown Candidate" else file.name.split(".")[0],
                "email": contact["email"],
                "phone": contact["phone"],
                "source": "Uploaded Resume",
                "parsed_resume": parsed_resume
            })

    if not candidates_pool:
        st.info("No candidates loaded. Upload some resumes or check 'Include Pre-loaded Talent Pool' to start matching.")
        return

    # Calculate match scores
    similarity_engine = SimilarityEngine()
    evaluated_candidates = []

    with st.spinner("Computing semantic match scores locally..."):
        for c in candidates_pool:
            scores = similarity_engine.calculate_scores(c["parsed_resume"], parsed_jd, weights=custom_weights)
            evaluated_candidates.append({
                "id": c["id"],
                "name": c["name"],
                "email": c["email"],
                "phone": c["phone"],
                "source": c["source"],
                "overall_score": scores["overall_score"],
                "skills_match": scores["skills_match"],
                "experience_match": scores["experience_match"],
                "education_match": scores["education_match"],
                "projects_match": scores["projects_match"],
                "certifications_match": scores["certifications_match"],
                "parsed_resume": c["parsed_resume"]
            })

    # Sort candidates by overall score
    evaluated_candidates.sort(key=lambda x: x["overall_score"], reverse=True)

    # 4. Candidate Ranking Table
    st.subheader("4. Candidate Match Rankings")
    
    # Create pandas dataframe for display
    df_data = []
    for rank, c in enumerate(evaluated_candidates, 1):
        df_data.append({
            "Rank": rank,
            "Name": c["name"],
            "Overall Match": f"{c['overall_score']:.1f}%",
            "Skills Match": f"{c['skills_match']:.1f}%",
            "Experience Match": f"{c['experience_match']:.1f}%",
            "Education Match": f"{c['education_match']:.1f}%",
            "Source": c["source"],
            "Email": c["email"],
            "Phone": c["phone"]
        })
    df = pd.DataFrame(df_data)
    st.dataframe(df.set_index("Rank"), use_container_width=True)

    # Export CSV Option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Export Evaluation Report (CSV)",
        csv,
        "evaluation_report.csv",
        "text/csv",
        key="download-csv"
    )

    st.markdown("---")

    # 5. Candidate Comparison
    st.subheader("5. Side-by-Side Candidate Comparison")
    st.markdown("Select 2 or 3 candidates to compare their strengths, matches, and recommendations:")

    candidate_names = [c["name"] for c in evaluated_candidates]
    selected_names = st.multiselect("Select Candidates to Compare", candidate_names, default=candidate_names[:2] if len(candidate_names) >= 2 else candidate_names[:1])

    if selected_names:
        selected_candidates = [c for c in evaluated_candidates if c["name"] in selected_names]
        
        # Display scores table
        st.markdown("### Match Score Comparison")
        render_comparison_matrix(selected_candidates, parsed_jd)

        # Plot radar or bar chart comparison
        st.markdown("### Semantic Match Profile")
        fig = go.Figure()
        categories = ['Skills', 'Experience', 'Education', 'Projects', 'Certifications']
        
        for c in selected_candidates:
            values = [
                c["skills_match"],
                c["experience_match"],
                c["education_match"],
                c["projects_match"],
                c["certifications_match"]
            ]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=c["name"]
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            height=350,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # AI-powered Comparison Insights
        if rag_pipeline.has_api_key():
            st.markdown("### AI Match Comparison Insights")
            if st.button("Generate AI Comparison Insights"):
                with st.spinner("Generating side-by-side AI analysis..."):
                    # We compare their relative fits. Create a summary of their profiles to send to LLM.
                    comparison_context = ""
                    for c in selected_candidates:
                        comparison_context += f"""
                        Candidate: {c['name']}
                        Scores: Overall={c['overall_score']:.1f}%, Skills={c['skills_match']:.1f}%, Experience={c['experience_match']:.1f}%
                        Extracted Experience: {c['parsed_resume'].get('experience', '')[:400]}...
                        Extracted Skills: {c['parsed_resume'].get('skills', '')[:300]}
                        ---
                        """
                        
                    prompt = f"""
                    You are an executive HR consultant. Provide a brief side-by-side comparative analysis of the following candidate fits for the target role:
                    Target JD Requirements:
                    - Skills: {parsed_jd.get('skills', '')}
                    - Experience: {parsed_jd.get('experience', '')}

                    Candidates Data:
                    {comparison_context}

                    Write a concise comparison detailing:
                    - Who is the strongest candidate and why.
                    - Key trade-offs between them (e.g. one has more experience, the other has better modern skills).
                    - Final recommendation for interview sequencing.
                    
                    Return your response in clean Markdown.
                    """
                    try:
                        res_text = rag_pipeline.generate_comparison(parsed_jd, selected_candidates)
                        st.markdown(res_text)
                    except Exception as e:
                        st.error(f"Error generating comparison insights: {e}")
        else:
            st.info("Input a Groq API Key in the sidebar to generate comparative text summaries.")
        
        # Deep Dive Analysis of Individual Candidates
        st.markdown("### Individual Profiles & AI Analysis")
        for c in selected_candidates:
            with st.expander(f"👤 Deep Dive: {c['name']} (Match: {c['overall_score']:.1f}%)"):
                col_c1, col_c2 = st.columns([1, 2])
                with col_c1:
                    st.write(f"**Email:** {c['email']}")
                    st.write(f"**Phone:** {c['phone']}")
                    st.write(f"**Source:** {c['source']}")
                    
                    render_glass_card("Skills Match", f"{c['skills_match']:.1f}%", status="info" if c['skills_match'] > 50 else "danger")
                    render_glass_card("Experience Match", f"{c['experience_match']:.1f}%", status="success" if c['experience_match'] > 75 else "warning")
                
                with col_c2:
                    st.markdown("#### Resume Sections")
                    tab_sk, tab_ex, tab_ed, tab_pr = st.tabs(["Skills", "Experience", "Education", "Projects"])
                    with tab_sk:
                        st.write(c["parsed_resume"].get("skills", "None found."))
                    with tab_ex:
                        st.write(c["parsed_resume"].get("experience", "None found."))
                    with tab_ed:
                        st.write(c["parsed_resume"].get("education", "None found."))
                    with tab_pr:
                        st.write(c["parsed_resume"].get("projects", "None found."))
                    

