import os, json
import math
import matplotlib.pyplot as plt

def split_to_buckets(objects, split_points):
    
    split_points = sorted(split_points)
    sp_len = len(split_points)
    bucket_data = {'bucket{}'.format(idx+1): list() for idx in range(sp_len+1)}
    
    for obj in objects:
        # rp = obj['relative_popularity']
        if obj['pageviews'] != 0:
            rp = math.log(int(obj['pageviews']), 10)
        else:
            rp = 0
        
        if rp < split_points[0]:
            if 'bucket1' in bucket_data.keys():
                bucket_data['bucket1'].append(obj)
            else:
                bucket_data['bucket1'] = [obj]
        
        if rp >= split_points[-1]:
            if 'bucket{}'.format(sp_len+1) in bucket_data.keys():
                bucket_data['bucket{}'.format(sp_len+1)].append(obj)
            else:
                bucket_data['bucket{}'.format(sp_len+1)] = [obj]

        for i in range(sp_len-1):
            if split_points[i] <= rp < split_points[i + 1]:
                if 'bucket{}'.format(i+2) in bucket_data.keys():
                    bucket_data['bucket{}'.format(i+2)].append(obj)
                else:
                    bucket_data['bucket{}'.format(i+2)] = [obj]
    return bucket_data

def plot_bucket_num(results_per_bk):
    bucket_num = []
    for bk_name, objs in results_per_bk.items():
        bucket_num.append(len(objs))
    
    plt.bar(range(len(bucket_num)), bucket_num)
    # plt.title("EQ test-set")
    plt.tight_layout()
    plt.show()

def calculate_accuracy(results_per_bk):
    acc_per_bk = {}
    for bk_name, bk_data in results_per_bk.items():
        acc = sum([1 if obj['is_correct'] else 0 for obj in bk_data]) / len(bk_data)
        acc_per_bk[bk_name] = acc
    return acc_per_bk


def main():
    result_dir = "component0_preprocessing/generated_data/popQA_EQformat/results"
    
    # result_filename = "22.opt-350m.bf_norag_results.jsonl"
    # result_filename = "22.opt-350m.bf_rag_results.jsonl"
    # result_filename = "22.opt-350m.af_norag_results.jsonl"
    # result_filename = "22.opt-350m.af_rag_results.jsonl"
    
    # result_filename = "182.opt-350m.bf_norag_results.jsonl"
    # result_filename = "182.opt-350m.bf_rag_results.jsonl"
    # result_filename = "182.opt-350m.af_norag_results.jsonl"
    # result_filename = "182.opt-350m.af_rag_results.jsonl"
    
    # result_filename = "106.opt-350m.bf_norag_results.jsonl"
    # result_filename = "106.opt-350m.bf_rag_results.jsonl"
    # result_filename = "106.opt-350m.af_norag_results.jsonl"
    # result_filename = "106.opt-350m.af_rag_results.jsonl"
    
    # result_filename = "91.opt-350m.bf_norag_results.jsonl"
    # result_filename = "91.opt-350m.bf_rag_results.jsonl"
    # result_filename = "91.opt-350m.af_norag_results.jsonl"
    # result_filename = "91.opt-350m.af_rag_results.jsonl"
    
    result_filename = "106_22_182.opt-350m.bf_norag_results.jsonl"
    result_filename = "106_22_182.opt-350m.bf_rag_results.jsonl"
    result_filename = "106_22_182.opt-350m.af_norag_results.jsonl"
    # result_filename = "106_22_182.opt-350m.af_rag_results.jsonl"
    
    result_file = os.path.join(result_dir, result_filename)
    split_points = [2, 3, 4, 5] # Good for my pageviews
    # split_points = [2, 3, 4, 5]
    
    with open(result_file, 'r') as file:
        results = [json.loads(line) for line in file]
    
    results_per_bk = split_to_buckets(results, split_points)
    # plot_bucket_num(results_per_bk)
    
    acc = calculate_accuracy({"all": results})
    print(acc)
    acc_per_bk = calculate_accuracy(results_per_bk)
    print(acc_per_bk)


if __name__ == "__main__":
    main()

