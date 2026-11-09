
import streamlit as st
from groq import Groq


# =====================================================
# GROQ CLIENT
# =====================================================

def get_groq_client():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from .streamlit/secrets.toml"
        )

    return Groq(api_key=api_key)


# =====================================================
# AI RESPONSE
# =====================================================

def get_ai_response(prompt):

    try:

        client = get_groq_client()

        model_name = st.secrets.get(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        response = client.chat.completions.create(

            model=model_name,

            messages=[
                {
                    "role": "system",
                    "content": """
You are NoteWise AI, an intelligent academic study assistant.

Your responsibilities:
- Help students understand their study material.
- Explain difficult concepts clearly.
- Generate accurate summaries.
- Identify important information.
- Prepare examination questions.
- Create topic-wise revision material.

Guidelines:
1. Use clear and structured explanations.
2. Use headings and bullet points.
3. Preserve technical terminology.
4. Give examples where useful.
5. Do not fabricate information from the student's notes.
6. Clearly distinguish information from the notes
   from additional explanations.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4,
            max_tokens=4000
        )

        answer = response.choices[0].message.content

        if not answer:
            return "No response received. Please try again."

        return answer

    except Exception as error:

        return "Groq API Error: " + str(error)


# =====================================================
# COMMON STUDY TASK
# =====================================================

def run_study_task(task, notes):

    if not notes or not notes.strip():

        return "Please provide study notes first."

    prompt = f"""
TASK:
{task}

STUDENT NOTES:
{notes}

INSTRUCTIONS:
- Base your answer primarily on the supplied notes.
- Organise your response using meaningful headings.
- Explain difficult concepts in student-friendly language.
- Do not invent information and attribute it to the notes.
- If the notes do not contain enough information,
  clearly mention this.
"""

    return get_ai_response(prompt)


# =====================================================
# SMART SUMMARY
# =====================================================

def generate_summary(notes):

    task = """
Create a detailed but concise academic summary.

Include:
1. Main topic and introduction.
2. Important concepts.
3. Explanations of key ideas.
4. Important definitions and terminology.
5. A short final revision summary.

Use clear headings and structured paragraphs.
"""

    return run_study_task(task, notes)


# =====================================================
# IMPORTANT INFORMATION
# =====================================================

def generate_important_info(notes):

    task = """
Extract important information from these notes.

Identify:
1. Important definitions.
2. Important facts.
3. Formulas and equations.
4. Dates, names and terminology.
5. Important examples.
6. Concepts that may be relevant for examinations.

Explain each item briefly.
"""

    return run_study_task(task, notes)


# =====================================================
# KEY POINTS
# =====================================================

def generate_key_points(notes):

    task = """
Extract the most important key points.

Requirements:
1. Use numbered headings.
2. Give concise bullet points.
3. Preserve technical terms.
4. Avoid unnecessary repetition.
5. Make the output useful for quick revision.
"""

    return run_study_task(task, notes)


# =====================================================
# QUIZ GENERATOR
# =====================================================

def generate_quiz(
    notes,
    number_of_questions=10,
    difficulty="Mixed"
):

    task = f"""
Generate {number_of_questions} multiple-choice questions.

Difficulty level: {difficulty}

Requirements:

1. Create questions based on the supplied notes.
2. Give four options for each question:
   A, B, C and D.
3. Include a mixture of conceptual and factual questions.
4. Avoid repeating questions.
5. After all questions, provide a separate answer key.
6. Give a short explanation for every correct answer.
7. Do not reveal answers immediately after individual questions.
"""

    return run_study_task(task, notes)


# =====================================================
# TOPIC-WISE REVISION
# =====================================================

def generate_revision_guide(notes):

    task = """
Create a topic-wise revision guide.

For each topic include:

1. Topic name.
2. Important subtopics.
3. Short explanation.
4. Important definitions.
5. Important facts or formulas.
6. Possible examination questions.
7. Quick revision points.

Organise the content logically.
Highlight the most important topics.
"""

    return run_study_task(task, notes)


# =====================================================
# ASK QUESTIONS ABOUT NOTES
# =====================================================

def ask_question(notes, question):

    if not question or not question.strip():

        return "Please enter a question."

    task = f"""
Answer the following student's question:

QUESTION:
{question}

Use the supplied study notes as your primary source.

Requirements:
1. Explain the answer clearly.
2. Use simple language where appropriate.
3. Give examples if useful.
4. Preserve technical terminology.
5. If the answer is not present in the notes,
   clearly state that.
"""

    return run_study_task(task, notes)


# =====================================================
# STUDY ASSISTANT INTERFACE
# =====================================================

def render_study_assistant(notes):

    st.markdown("## AI Study Assistant")

    st.caption(
        "Understand your notes, prepare for examinations "
        "and revise more effectively."
    )

    if not notes or not notes.strip():

        st.info(
            "Digitize your notes first or provide study material."
        )

        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Smart Summary",
        "Important Info",
        "Key Points",
        "Quiz",
        "Topic Revision",
        "Ask Notes"
    ])

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    with tab1:

        st.subheader("Smart Summary")

        st.write(
            "Generate a structured summary of your notes."
        )

        if st.button(
            "Generate Summary",
            key="sa_summary",
            use_container_width=True
        ):

            with st.spinner("Preparing summary..."):

                st.session_state.sa_result = generate_summary(
                    notes
                )

                st.session_state.sa_result_title = "Smart Summary"

        if st.session_state.get("sa_result"):

            st.markdown(
                st.session_state.sa_result
            )

    # -------------------------------------------------
    # IMPORTANT INFORMATION
    # -------------------------------------------------

    with tab2:

        st.subheader("Important Information")

        st.write(
            "Identify important facts, definitions and formulas."
        )

        if st.button(
            "Find Important Information",
            key="sa_important",
            use_container_width=True
        ):

            with st.spinner("Finding important information..."):

                st.session_state.sa_result = generate_important_info(
                    notes
                )

                st.session_state.sa_result_title = "Important Information"

        if st.session_state.get("sa_result"):

            st.markdown(
                st.session_state.sa_result
            )

    # -------------------------------------------------
    # KEY POINTS
    # -------------------------------------------------

    with tab3:

        st.subheader("Key Points")

        st.write(
            "Convert lengthy notes into quick revision points."
        )

        if st.button(
            "Extract Key Points",
            key="sa_keypoints",
            use_container_width=True
        ):

            with st.spinner("Extracting key points..."):

                st.session_state.sa_result = generate_key_points(
                    notes
                )

                st.session_state.sa_result_title = "Key Points"

        if st.session_state.get("sa_result"):

            st.markdown(
                st.session_state.sa_result
            )

    # -------------------------------------------------
    # QUIZ
    # -------------------------------------------------

    with tab4:

        st.subheader("Quiz Generator")

        number_of_questions = st.slider(
            "Number of questions",
            min_value=5,
            max_value=20,
            value=10,
            key="sa_question_count"
        )

        difficulty = st.selectbox(
            "Difficulty level",
            [
                "Easy",
                "Medium",
                "Hard",
                "Mixed"
            ],
            key="sa_difficulty"
        )

        if st.button(
            "Generate Quiz",
            key="sa_quiz",
            use_container_width=True
        ):

            with st.spinner("Generating quiz..."):

                st.session_state.sa_result = generate_quiz(
                    notes,
                    number_of_questions,
                    difficulty
                )

                st.session_state.sa_result_title = "Quiz"

        if st.session_state.get("sa_result"):

            st.markdown(
                st.session_state.sa_result
            )

    # -------------------------------------------------
    # TOPIC REVISION
    # -------------------------------------------------

    with tab5:

        st.subheader("Topic-wise Revision")

        st.write(
            "Organise your notes into topics and subtopics."
        )

        if st.button(
            "Create Revision Guide",
            key="sa_revision",
            use_container_width=True
        ):

            with st.spinner("Preparing revision guide..."):

                st.session_state.sa_result = generate_revision_guide(
                    notes
                )

                st.session_state.sa_result_title = "Topic-wise Revision"

        if st.session_state.get("sa_result"):

            st.markdown(
                st.session_state.sa_result
            )

    # -------------------------------------------------
    # ASK NOTES
    # -------------------------------------------------

    with tab6:

        st.subheader("Ask Questions About Your Notes")

        question = st.text_input(
            "Enter your question",
            placeholder="Explain this concept in simple words...",
            key="sa_question"
        )

        if st.button(
            "Ask NoteWise AI",
            key="sa_ask",
            use_container_width=True
        ):

            if not question.strip():

                st.warning("Please enter a question.")

            else:

                with st.spinner("Preparing your answer..."):

                    st.session_state.sa_result = ask_question(
                        notes,
                        question
                    )

                    st.session_state.sa_result_title = "Answer"

        if st.session_state.get("sa_result"):

            st.markdown(
                st.session_state.sa_result
            )
