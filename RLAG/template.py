class Template:
    
    en_usmle_instruction = "You are a medical student and need to answer the medical questions based on your medical knowledge. Here are the medical questions you entered and give your answers directly: "
    
    en_usmle_sample_without_ctx_prompt_prefix = lambda que, options: f"""You are an AI that answers single-choice questions by selecting one of the provided options. Given the question and options separated by semicolons (;), output only one the exact text of the correct option. Do not include any additional text, explanations, or multiple options.
    <Example>Question: What is the capital of France?
    Options: Berlin; Madrid; Paris; Rome
    Answer: Paris<\Example>
    Now, answer the following question:
    Question: {que}
    Options: {options}
    Answer: """
    
    en_usmle_sample_with_ctx_prompt_prefix = lambda ctx, que, options: f"""You are an AI that answers single-choice questions by selecting one of the provided options. Given the question and options separated by semicolons (;), output only one the exact text of the correct option. Do not include any additional text, explanations, or multiple options.
    <Example>Question: What is the capital of France?
    Options: Berlin; Madrid; Paris; Rome
    Answer: Paris<\Example>
    Now, answer the following question:
    Related literature: {ctx}
    Question: {que}
    Options: {options}
    Answer: """
    
    en_usmle_rag_input_prompt_prefix = lambda ctx, que, options: f"Provide a direct answer to the following medical question without analyzing or explaining your answer.\nCTX: {ctx}\n{que}\n{options}"
    en_usmle_ori_input_prompt_prefix = lambda que, options: f"Provide a direct answer to the following medical question without analyzing or explaining your answer.\n{que}\n{options}"
    
    logprob_sample_ctx_template = lambda ctx, que: f"Provide a direct answer to the following medical question without analyzing or explaining your answer.\nRelated literature: {ctx}\nQuestion: {que}"
    logprob_sample_ori_template = lambda que: f"Provide a direct answer to the following medical question without analyzing or explaining your answer.\nQuestion: {que}"
    
    en_ctx_process_template = "Input a medical literature fragment, process it into more independent medical knowledge points while keeping the original content as much as possible, and directly output a sentence content. All contents are kept in one line, and line breaks and special symbols cannot be output: "
    
    CoT_Instruction = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
    The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., 
    <think> reasoning process here </think> 
    <answer> answer here </answer>.'''
    
    CoT_rag_temp = lambda ctx, que, options: f'''Your Inside Knowledge: {ctx}\nQuestion: {que}\noptions: {options}'''
    CoT_ori_temp = lambda que, options: f'''Question: {que}\noptions: {options}'''
    
    def __init__(self):
        pass