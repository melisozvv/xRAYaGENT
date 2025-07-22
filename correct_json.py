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
            if isinstance(question_data["results"]["answer"], str):
                data[sample_id][question_id]["results"]= json.loads(question_data["results"]["answer"])
            elif isinstance(question_data["results"]["answer"], list):
                data[sample_id][question_id]["results"] = json.dumps(question_data["results"]["answer"][0])
            elif isinstance(question_data["results"]["answer"], dict):
                data[sample_id][question_id]["results"] = question_data["results"]["answer"]
            else:
                print(type(question_data["results"]["answer"]))

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
        
if __name__ == "__main__":
    json_file = "/home/xiz569/rajpurkarlab/home/xiz569/melis/xRAYaGENT/output/xray_analysis_results_0723.json"
    output_file = "/home/xiz569/rajpurkarlab/home/xiz569/melis/xRAYaGENT/output/xray_analysis_results_0722.json"
    correct_json(json_file, output_file)
        



