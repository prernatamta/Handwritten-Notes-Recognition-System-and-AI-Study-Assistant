import streamlit as st
import json
import re
from groq import Groq

def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from .streamlit/secrets.toml"
        )
    return Groq(api_key=api_key)

def get_ai_response(prompt):
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": """
You are Scriptify, an intelligent academic study assistant.

Help students understand study material, generate summaries,
identify important information, prepare examination questions
and create revision material.

Use clear headings, structured explanations and examples.
Do not fabricate information from the student's notes.
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
- Organise your answer using meaningful headings.
- Explain difficult concepts in student-friendly language.
- Do not invent information and attribute it to the notes.
- If the notes do not contain enough information,
  clearly mention this.
"""

    return get_ai_response(prompt)

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
# QUIZ GENERATOR - STRUCTURED JSON
# =====================================================

def generate_quiz(notes, number_of_questions=10, difficulty="Mixed"):

    if not notes or not notes.strip():
        return {"error": "Please provide study notes first."}

    prompt = f"""
Create exactly {number_of_questions} multiple-choice questions
based ONLY on the student's notes below.

Difficulty: {difficulty}

Return ONLY valid JSON. Do not include Markdown,
code fences, introductions or text outside the JSON.

Use this exact structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": {{
                "A": "First option",
                "B": "Second option",
                "C": "Third option",
                "D": "Fourth option"
            }},
            "answer": "A",
            "explanation": "Explanation of the correct answer"
        }}
    ]
}}

IMPORTANT:
- Generate exactly {number_of_questions} questions.
- Every question must have exactly four options.
- Option keys must be A, B, C and D.
- The answer must be only A, B, C or D.
- There must be exactly one correct answer per question.
- Include conceptual and factual questions.
- Avoid repeated questions.
- Keep all questions relevant to the supplied notes.
- Explanations should be short and educational.
- Do not reveal the answer in the question or options.

STUDENT NOTES:
{notes}
"""

    try:
        response = get_ai_response(prompt)

        if response.startswith("Groq API Error:"):
            return {"error": response}

        # Remove possible Markdown code fences
        cleaned_response = response.strip()
        cleaned_response = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned_response,
            flags=re.IGNORECASE
        )
        cleaned_response = re.sub(
            r"\s*```$",
            "",
            cleaned_response
        )

        # Extract JSON if extra text was returned
        start = cleaned_response.find("{")
        end = cleaned_response.rfind("}")

        if start == -1 or end == -1:
            return {
                "error": "The AI did not return valid quiz data. Please generate again."
            }

        quiz_data = json.loads(cleaned_response[start:end + 1])

        questions = quiz_data.get("questions", [])

        if not questions:
            return {"error": "No questions were generated. Please try again."}

        valid_questions = []

        for question in questions:

            options = question.get("options", {})
            answer = str(question.get("answer", "")).upper()

            if (
                not question.get("question")
                or not isinstance(options, dict)
                or not all(key in options for key in ["A", "B", "C", "D"])
                or answer not in ["A", "B", "C", "D"]
            ):
                continue

            valid_questions.append({
                "question": question["question"],
                "options": {
                    "A": str(options["A"]),
                    "B": str(options["B"]),
                    "C": str(options["C"]),
                    "D": str(options["D"])
                },
                "answer": answer,
                "explanation": str(
                    question.get("explanation", "No explanation provided.")
                )
            })

        if not valid_questions:
            return {"error": "The generated quiz had an invalid format. Please try again."}

        return {"questions": valid_questions}

    except json.JSONDecodeError:
        return {"error": "Could not read the generated quiz. Please generate it again."}

    except Exception as error:
        return {"error": "Quiz generation error: " + str(error)}


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
# QUIZ INTERFACE
# =====================================================

def render_interactive_quiz():

    quiz_data = st.session_state.get("sa_quiz_data")

    if not quiz_data:
        return

    if "error" in quiz_data:
        st.error(quiz_data["error"])
        return

    questions = quiz_data["questions"]

    if not questions:
        st.warning("No questions are available.")
        return

    st.markdown("---")
    st.subheader("Your Quiz")

    st.caption(
        f"{len(questions)} questions | Select one answer for each question."
    )

    submitted = st.session_state.get("sa_quiz_submitted", False)

    selected_answers = {}

    with st.form("sa_interactive_quiz_form"):

        for index, question in enumerate(questions):

            st.markdown(f"### Question {index + 1}")

            st.write(question["question"])

            option_keys = ["A", "B", "C", "D"]

            selected = st.radio(
                "Choose your answer:",
                options=option_keys,
                format_func=lambda key, opts=question["options"]: (
                    f"{key}. {opts[key]}"
                ),
                index=None,
                key=f"sa_quiz_option_{index}",
                disabled=submitted,
                label_visibility="collapsed"
            )

            selected_answers[index] = selected

            st.markdown("---")

        submit_quiz = st.form_submit_button(
            "Submit Quiz",
            use_container_width=True,
            disabled=submitted
        )

    if submit_quiz:

        unanswered = [
            index for index, answer in selected_answers.items()
            if answer is None
        ]

        if unanswered:
            st.warning(
                f"You have {len(unanswered)} unanswered question(s). "
                "You can still submit your quiz."
            )

        st.session_state.sa_quiz_answers = selected_answers
        st.session_state.sa_quiz_submitted = True
        st.rerun()

    # -------------------------------------------------
    # RESULTS
    # -------------------------------------------------

    if st.session_state.get("sa_quiz_submitted", False):

        answers = st.session_state.sa_quiz_answers

        score = 0

        for index, question in enumerate(questions):

            if answers.get(index) == question["answer"]:
                score += 1

        total = len(questions)
        percentage = (score / total) * 100

        st.markdown("---")
        st.subheader("Quiz Results")

        st.metric(
            "Your Score",
            f"{score} / {total}"
        )

        st.progress(percentage / 100)

        st.write(f"**Percentage:** {percentage:.1f}%")

        if percentage >= 80:
            st.success("Excellent work! You have a strong understanding of these notes.")

        elif percentage >= 50:
            st.info("Good effort! Review the explanations to improve further.")

        else:
            st.warning("Keep practising. Review your notes and try again.")

        st.markdown("---")
        st.subheader("Answer Review")

        for index, question in enumerate(questions):

            user_answer = answers.get(index)
            correct_answer = question["answer"]

            st.markdown(f"### Question {index + 1}")
            st.write(question["question"])

            if user_answer == correct_answer:
                st.success(
                    f"Correct! Your answer: {user_answer}"
                )
            else:
                st.error(
                    f"Your answer: {user_answer if user_answer else 'Not answered'}"
                )

                st.info(
                    f"Correct answer: {correct_answer}. "
                    f"{question['options'][correct_answer]}"
                )

            st.write(
                "**Explanation:** " + question["explanation"]
            )

            st.markdown("---")

        if st.button(
            "Retake Quiz",
            key="sa_retake_quiz",
            use_container_width=True
        ):

            st.session_state.sa_quiz_answers = {}
            st.session_state.sa_quiz_submitted = False

            for index in range(len(questions)):
                key = f"sa_quiz_option_{index}"
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun()


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

    # Initialise session state
    if "sa_result" not in st.session_state:
        st.session_state.sa_result = ""

    if "sa_quiz_data" not in st.session_state:
        st.session_state.sa_quiz_data = None

    if "sa_quiz_answers" not in st.session_state:
        st.session_state.sa_quiz_answers = {}

    if "sa_quiz_submitted" not in st.session_state:
        st.session_state.sa_quiz_submitted = False

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
        st.write("Generate a structured summary of your notes.")

        if st.button(
            "Generate Summary",
            key="sa_summary",
            use_container_width=True
        ):

            with st.spinner("Preparing summary..."):
                st.session_state.sa_result = generate_summary(notes)

        if st.session_state.sa_result:
            st.markdown(st.session_state.sa_result)

    # -------------------------------------------------
    # IMPORTANT INFORMATION
    # -------------------------------------------------

    with tab2:

        st.subheader("Important Information")
        st.write("Identify important facts, definitions and formulas.")

        if st.button(
            "Find Important Information",
            key="sa_important",
            use_container_width=True
        ):

            with st.spinner("Finding important information..."):
                st.session_state.sa_result = generate_important_info(notes)

        if st.session_state.sa_result:
            st.markdown(st.session_state.sa_result)

    # -------------------------------------------------
    # KEY POINTS
    # -------------------------------------------------

    with tab3:

        st.subheader("Key Points")
        st.write("Convert lengthy notes into quick revision points.")

        if st.button(
            "Extract Key Points",
            key="sa_keypoints",
            use_container_width=True
        ):

            with st.spinner("Extracting key points..."):
                st.session_state.sa_result = generate_key_points(notes)

        if st.session_state.sa_result:
            st.markdown(st.session_state.sa_result)

    # -------------------------------------------------
    # QUIZ
    # -------------------------------------------------

    with tab4:

        st.subheader("Quiz Generator")

        st.write(
            "Test your understanding with an interactive quiz "
            "based on your study material."
        )

        number_of_questions = st.slider(
            "Number of questions",
            min_value=5,
            max_value=20,
            value=10,
            key="sa_question_count"
        )

        difficulty = st.selectbox(
            "Difficulty level",
            ["Easy", "Medium", "Hard", "Mixed"],
            key="sa_difficulty"
        )

        if st.button(
            "Generate Quiz",
            key="sa_quiz",
            use_container_width=True
        ):

            with st.spinner("Generating your interactive quiz..."):

                st.session_state.sa_quiz_data = generate_quiz(
                    notes,
                    number_of_questions,
                    difficulty
                )

                st.session_state.sa_quiz_answers = {}
                st.session_state.sa_quiz_submitted = False

                for index in range(20):
                    key = f"sa_quiz_option_{index}"
                    if key in st.session_state:
                        del st.session_state[key]

        render_interactive_quiz()

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
                st.session_state.sa_result = generate_revision_guide(notes)

        if st.session_state.sa_result:
            st.markdown(st.session_state.sa_result)

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
            "Ask Scriptify",
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

        if st.session_state.sa_result:
            st.markdown(st.session_state.sa_result)
