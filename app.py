import streamlit as st
import pandas as pd
import re
import openai

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit UI
st.title("LLM Evaluation Tool")
st.write("Upload an Excel file for processing. The expected formats are:")
st.write("1. Columns: Index, Question, Context, Answer, Reference Context, Reference Answer")
st.write("2. Columns: Index, Conversation, Agent Prompt")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "csv"])

if uploaded_file:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        if "Question" in df.columns and "Context" in df.columns and "Answer" in df.columns:
            # Code 1 processing
            required_columns = ["Index", "Question", "Context", "Answer", "Reference Context", "Reference Answer"]
            if not all(col in df.columns for col in required_columns):
                st.error(f"The uploaded file must contain these columns: {', '.join(required_columns)}.")
            else:
                st.write("Preview of Uploaded Data:")
                st.dataframe(df.head())

                num_metrics = st.number_input("Enter the number of metrics you want to define:", min_value=1, step=1)

                if "combined_results" not in st.session_state:
                    st.session_state.combined_results = []

                for i in range(num_metrics):
                    st.markdown(f"""
                        <hr style="border: 5px solid #000000;">
                        <h3 style="background-color: #f0f0f0; padding: 10px; border: 2px solid #000000;">
                            Metric {i + 1}
                        </h3>
                    """, unsafe_allow_html=True)

                    selected_columns = st.multiselect(
                        f"Select columns for Metric {i + 1}:",
                        options=required_columns[1:],
                        key=f"columns_{i}"
                    )

                    toggle_prompt = st.checkbox(
                        f"Automatically generate system prompt for Metric {i + 1}", key=f"toggle_prompt_{i}"
                    )

                    if toggle_prompt:
                        if i % 2 == 0:
                            system_prompt = """You are a RELEVANCE grader; providing the relevance of the given question to the given answer.
                            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant."""
                        else:
                            system_prompt = """You are a FACTUAL ACCURACY grader; evaluating the factual correctness of the given answer based on the question and context.
                            Respond only as a number from 0 to 10 where 0 indicates completely factually inaccurate and 10 indicates completely factually accurate."""

                        st.text_area(
                            f"Generated System Prompt for Metric {i + 1}:",
                            value=system_prompt,
                            height=200
                        )
                    else:
                        system_prompt = st.text_area(
                            f"Enter the System Prompt for Metric {i + 1}:",
                            height=200
                        )

                    if st.button(f"Metric {i + 1} Results", key=f"generate_results_{i}"):
                        column_mapping = {
                            "Question": "question",
                            "Context": "context",
                            "Answer": "answer",
                            "Reference Context": "reference_context",
                            "Reference Answer": "reference_answer"
                        }
                        results = []
                        for index, row in df.iterrows():
                            params = {"system_prompt": system_prompt}
                            for col in selected_columns:
                                if col in column_mapping:
                                    params[column_mapping[col]] = row[col]

                            try:
                                # Constructing the evaluation prompt with system prompt and column values
                                evaluation_prompt = f"""
                                {system_prompt}

                                Below is the data for evaluation:
                                {''.join([f'{col}: {row[col]}\n' for col in selected_columns])}

                                Based on the provided data, evaluate the following:
                                1. **Criteria**: Explain how the evaluation is derived.
                                2. **Supporting Evidence**: Provide specific examples from the data.
                                3. **Score**: Provide a numerical or qualitative score.
                                """

                                response = openai.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": "system", "content": "You are an evaluator analyzing the provided data."},
                                        {"role": "user", "content": evaluation_prompt}
                                    ]
                                )
                                response_content = response.choices[0].message.content.strip()
                                st.write(response_content)
                                # Parsing the GPT response for Criteria, Supporting Evidence, and Score
                                # Parsing the GPT response for Criteria, Supporting Evidence, and Score
                                criteria_match = re.search(r"1\.\s*Criteria:\s*(.*?)(?=\n2\.)", response_content, re.S)
                                evidence_match = re.search(r"2\.\s*Supporting Evidence:\s*(.*?)(?=\n3\.)", response_content, re.S)
                                score_match = re.search(r"3\.\s*Score:\s*(.*)", response_content)

                                criteria = criteria_match.group(1).strip() if criteria_match else "Not available"
                                evidence = evidence_match.group(1).strip() if evidence_match else "Not available"
                                score = score_match.group(1).strip() if score_match else "Not available"


                                result_row = {
                                    "Index": row["Index"],
                                    "Metric": f"Metric {i + 1}",
                                    "Selected Columns": ", ".join(selected_columns),
                                    "Score": score,
                                    "Criteria": criteria,
                                    "Supporting Evidence": evidence,
                                    "Question": row["Question"],
                                    "Context": row["Context"],
                                    "Answer": row["Answer"],
                                    "Reference Context": row["Reference Context"],
                                    "Reference Answer": row["Reference Answer"]
                                }
                                results.append(result_row)
                            except Exception as e:
                                results.append({
                                    "Index": row["Index"],
                                    "Metric": f"Metric {i + 1}",
                                    "Selected Columns": ", ".join(selected_columns),
                                    "Score": "Error",
                                    "Criteria": "Error",
                                    "Supporting Evidence": "Error",
                                    "Question": row["Question"],
                                    "Context": row["Context"],
                                    "Answer": row["Answer"],
                                    "Reference Context": row["Reference Context"],
                                    "Reference Answer": row["Reference Answer"],
                                    "Error": str(e)
                                })

                        st.session_state.combined_results.extend(results)
                        st.write(f"Results for Metric {i + 1}:")
                        st.dataframe(pd.DataFrame(results))

                if num_metrics > 1 and st.button("Overall Results"):
                    if st.session_state.combined_results:
                        st.write("Combined Results:")
                        st.dataframe(pd.DataFrame(st.session_state.combined_results))
                    else:
                        st.warning("No results to combine. Please generate results for individual metrics first.")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
