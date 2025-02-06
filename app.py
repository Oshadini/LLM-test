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
                                            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 
                                        
                                            A few additional scoring guidelines:
                                            - Long answer should score equally well as short answer.
                                            - RELEVANCE score should increase as the answer provides more RELEVANT context to the question.
                                            - RELEVANCE score should increase as the answer provides RELEVANT context to more parts of the question.
                                            - Answer that is RELEVANT to some of the question should score of 2, 3, or 4. Higher score indicates more RELEVANCE.
                                            - Answer that is RELEVANT to most of the question should get a score of 5, 6, 7, or 8. Higher score indicates more RELEVANCE.
                                            - Answer that is RELEVANT to the entire question should get a score of 9 or 10. Higher score indicates more RELEVANCE.
                                            - Answer must be relevant and helpful for answering the entire question to get a score of 10.
                                            - Never elaborate."""
                        else:
                            system_prompt = """You are a FACTUAL ACCURACY grader; evaluating the factual correctness of the given answer based on the question and context.  
                                                Respond only as a number from 0 to 10 where 0 indicates completely factually inaccurate and 10 indicates completely factually accurate.  
                                                
                                                A few additional scoring guidelines:  
                                                - Long answers should score equally well as short answers if they are factually accurate.  
                                                - The FACTUAL ACCURACY score should increase as the answer contains more factually correct information related to the question and context.  
                                                - The presence of minor factual inaccuracies should lead to scores of 2, 3, or 4. Higher scores indicate fewer inaccuracies.  
                                                - If most parts of the answers are factually correct, the score should be 5, 6, 7, or 8. Higher scores indicate greater factual accuracy.  
                                                - If the entire answer is factually accurate and aligned with the question and context, the score should be 9 or 10.  
                                                - The answer must strictly avoid fabrications or contradictions to achieve a score of 10.  
                                                - Never elaborate."""

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
                                                                # Constructing the evaluation prompt with system prompt and column values
                                evaluation_prompt = f"""
                                {system_prompt}

                                Below is the data for evaluation:
                                {''.join([f'{col}: {row[col]}\n' for col in selected_columns])}

                                Based on the provided data, evaluate the following in this exact format:
                                1. Criteria: [Provide a detailed explanation of how the evaluation is derived.]
                                2. Supporting Evidence: [Provide specific examples from the data supporting the evaluation.]
                                3. Score: [Provide a numerical or qualitative score.]

                                Ensure the response strictly follows this format with numbered headings.
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

        #elif "Conversation" in df.columns and "Agent Prompt" in df.columns:
        # Agentic Testing (Code 2 processing with updates for lengthy prompts)
        elif "Conversation" in df.columns and "Agent Prompt" in df.columns:
            # Code 2 processing
            required_columns = ["Index", "Conversation", "Agent Prompt"]
            if not all(col in df.columns for col in required_columns):
                st.error(f"The uploaded file must contain these columns: {', '.join(required_columns)}.")
            else:
                st.write("Preview of Uploaded Data:")
                st.dataframe(df.head())
                
                MAX_PROMPT_LENGTH = 2000  # Define maximum allowable characters for the system prompt
                
                def truncate_prompt(prompt: str, max_length: int = MAX_PROMPT_LENGTH) -> str:
                    """
                    Truncate the prompt to fit within the allowed maximum length.
                    """
                    if len(prompt) > max_length:
                        return prompt[:max_length] + "..."
                    return prompt
                
                # Define function to evaluate conversation using GPT-4
                def evaluate_conversation(system_prompt: str, selected_columns: list, conversation: pd.DataFrame, metric_name: str) -> list:
                    """
                    Evaluate the conversation using GPT-4 based on the system prompt provided by the user.
                    """
                    results = []
                    
                    # Truncate the system prompt if it's too long
                    if len(system_prompt) > MAX_PROMPT_LENGTH:
                        st.warning(f"The system prompt exceeds {MAX_PROMPT_LENGTH} characters and will be truncated.")
                        system_prompt = truncate_prompt(system_prompt)
                    
                    for index, row in conversation.iterrows():
                        try:
                            # Construct the evaluation prompt for GPT-4
                            evaluation_prompt = f"""
                            System Prompt: {system_prompt}
                
                            Index: {row['Index']}
                            Conversation: {row['Conversation']}
                            Agent Prompt: {row['Agent Prompt']}
                
                            Evaluate the entire conversation for Agent-Goal Accuracy. Use the following format:
                            
                            Criteria: [Explain how well the Agent responded to the User's input and fulfilled their goals]
                            Supporting Evidence: [Highlight specific faulty or insufficient responses from the Agent]
                            Score: [Provide a numerical or qualitative score here]
                            """
                
                            # Call GPT-4 API
                            completion = openai.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": "You are an evaluator analyzing agent conversations."},
                                    {"role": "user", "content": evaluation_prompt}
                                ]
                            )
                
                            response_content = completion.choices[0].message.content.strip()
                
                            # Parse GPT-4 response into structured format
                            parsed_response = {
                                "Index": row["Index"],
                                "Metric": metric_name,
                                "Selected Columns": ", ".join(selected_columns),
                                "Score": "",
                                "Criteria": "",
                                "Supporting Evidence": "",
                                "Agent Prompt": row.get("Agent Prompt", ""),
                                "Conversation": row.get("Conversation", "")
                            }
                
                            # Extract values for Criteria, Supporting Evidence, and Score
                            for line in response_content.split("\n"):
                                line = line.strip()
                                if line.startswith("Criteria:"):
                                    parsed_response["Criteria"] = line.replace("Criteria:", "").strip()
                                elif line.startswith("Supporting Evidence:"):
                                    parsed_response["Supporting Evidence"] = line.replace("Supporting Evidence:", "").strip()
                                elif line.startswith("Score:"):
                                    parsed_response["Score"] = line.replace("Score:", "").strip()
                
                            # Validate extracted structure
                            if not (parsed_response["Criteria"] and parsed_response["Supporting Evidence"] and parsed_response["Score"]):
                                raise ValueError("Response does not contain the required structured fields.")
                
                            results.append(parsed_response)
                
                        except Exception as e:
                            results.append({
                                "Index": row["Index"],
                                "Metric": metric_name,
                                "Selected Columns": ", ".join(selected_columns),
                                "Score": "N/A",
                                "Criteria": "Error",
                                "Supporting Evidence": f"Error processing conversation: {e}",
                                "Agent Prompt": row.get("Agent Prompt", ""),
                                "Conversation": row.get("Conversation", "")
                            })
                
                    return results

                num_metrics = st.number_input("Enter the number of metrics you want to define:", min_value=1, step=1)
        
                if "system_prompts" not in st.session_state:
                    st.session_state.system_prompts = {}
                if "combined_results" not in st.session_state:
                    st.session_state.combined_results = []

                for i in range(num_metrics):
                    st.markdown(f"""
                        <hr style="border: 5px solid #000000;">
                        <h3 style="background-color: #f0f0f0; padding: 10px; border: 2px solid #000000;">
                            Metric {i + 1}
                        </h3>
                    """, unsafe_allow_html=True)

                    # Column selection remains unchanged
                    selected_columns = st.multiselect(
                        f"Select columns for Metric {i + 1}:",
                        options=required_columns[1:],  # Skip the Index column
                        key=f"columns_{i}"
                    )

                    toggle_prompt = st.checkbox(
                        f"Automatically generate system prompt for Metric {i + 1}", key=f"toggle_prompt_{i}"
                    )



                    if toggle_prompt:
               
                      system_prompt = """Role: You are responsible for evaluating the AGENT-GOAL ACCURACY of a conversation based on its alignment with the AGENT PROMPT.  

                                        Scoring Scale (0-10):  
                                        0 indicates the responses are entirely unrelated to the AGENT PROMPT.  
                                        1 to 4 reflects limited alignment, lacking depth, precision, or relevance. Scores of 1 to 2 represent negligible coverage with minimal helpfulness, while 3 to 4 indicate partial coverage with insufficient detail or completeness.  
                                        5 to 8 represents reasonable alignment with clear and relevant responses. Scores of 5 to 6 cover most of the prompt but have significant gaps, while 7 to 8 reflect strong alignment with minor omissions or gaps.  
                                        9 to 10 signifies comprehensive alignment with precise, complete, and entirely relevant responses. A score of 9 is near-perfect with minimal imperfections, and a score of 10 is fully sufficient and flawlessly aligned.  
                                        
                                        Criteria for evaluation include how well the agent responses address user inputs and the AGENT PROMPT, the accuracy, relevance, and helpfulness of the responses, and the completeness and precision in meeting the context of the AGENT PROMPT.  
                                        
                                        Supporting evidence should highlight areas of strong alignment and fulfillment of goals while identifying specific faults, such as misunderstanding, inaccuracy, or incompleteness, that detract from the relevance or helpfulness of the responses.  
                                        
                                        Output must be a single numerical score between 0 and 10 with no additional text.
                                        """
        

                      st.text_area(
                        f"Generated System Prompt for Metric {i + 1}:", value=system_prompt, height=200
                        )
                      st.success(f"System Prompt for Metric {i + 1} is Generated")

                    else:
                        system_prompt = st.text_area(
                            f"Enter the System Prompt for Metric {i + 1}:",
                            height=200
                        )
                    

                    # Generate results for each metric
                    if st.button(f"Metric {i + 1} Results", key=f"generate_results_{i}"):
                        if system_prompt.strip() == "":
                            st.error("Please enter a valid system prompt.")
                        else:
                            st.write("Evaluating conversations. Please wait...")

                            results = evaluate_conversation(system_prompt, selected_columns, df, f"Metric {i + 1}")
                            st.session_state.combined_results.extend(results)
                            st.write(f"Results for Metric {i + 1}:")
                            st.dataframe(pd.DataFrame(results))

                # Combine results for all metrics
                # Check if there are combined results before displaying them
                if num_metrics > 1 and st.button("Overall Results"):
                    try:
                        if st.session_state.combined_results:
                            st.write("Combined Results:")
                            st.dataframe(pd.DataFrame(st.session_state.combined_results))
                        else:
                            st.warning("No results to combine. Please generate results for individual metrics first.")
                    except Exception as e:
                        st.error(f"Error displaying combined results: {e}")


    
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
