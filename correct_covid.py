import os
import json

def correct_json(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for sample_id, sample_data in data.items():
        for question_id, question_data in sample_data.items():
            if "results" not in question_data:
                continue
            if "answer" not in question_data["results"]:
                continue
            try:
                if isinstance(question_data["results"]["answer"], str):
                    data[sample_id][question_id]["results"]["answer"] = json.loads(question_data["results"]["answer"])
                elif isinstance(question_data["results"]["answer"], list) and len(question_data["results"]["answer"]) > 0:
                    data[sample_id][question_id]["results"]["answer"] = json.dumps(question_data["results"]["answer"][0])
                elif isinstance(question_data["results"]["answer"], dict):
                    data[sample_id][question_id]["results"]["answer"] = question_data["results"]["answer"]
                else:
                    data[sample_id][question_id]["results"]["answer"] = question_data["results"]["answer"]
                    print(type(data[sample_id][question_id]["results"]["answer"]))
            except:
                print(f"Error in {sample_id} {question_id}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# read json fıle /Users/melisozvardar/Desktop/xRAYaGENT/output_google/xray_analysis_results.json and /Users/melisozvardar/Desktop/xRAYaGENT/output/xray_analysis_results_0723.json

with open('/Users/melisozvardar/Desktop/xRAYaGENT/output_google/xray_analysis_results.json', 'r') as f:
    google_data = json.load(f)

with open('/Users/melisozvardar/Desktop/xRAYaGENT/output/xray_analysis_results_0723.json', 'r') as f:
    agent_data = json.load(f)


save_data = agent_data.copy()
for sample_id, sample_data in agent_data.items():
    agent_data_question4 = sample_data['question4']
    google_data_question1 = google_data[sample_id]['question1']
    save_data[sample_id]['question4'] = google_data_question1

# save the data to a json file
with open('/Users/melisozvardar/Desktop/xRAYaGENT/output/xray_analysis_results_0722.json', 'w') as f:
    json.dump(save_data, f, indent=4)

correct_json('/Users/melisozvardar/Desktop/xRAYaGENT/output/xray_analysis_results_0722.json', '/Users/melisozvardar/Desktop/xRAYaGENT/output/xray_analysis_results_0722.json')
