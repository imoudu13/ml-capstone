from helpers import process_and_compare_pdfs
# Output differences
import json
import new

sample_1_new = "../policy-sample-1-new.pdf"
sample_1_expiring = "../policy-sample-1-expiring.pdf"
sample_2_new = " ../policy-sample-2-new.pdf"
sample_2_expiring = "../policy-sample-2-expiring.pdf"
my_sample_1 = "../my-sample-1.pdf"
my_sample_2 = "../my-sample-2.pdf"
reduced_1 = "sample_policy_1.pdf"
reduced_2 = "sample_policy_2.pdf"
# result = process_and_compare_pdfs(sample_1_new, sample_1_expiring)
result1 = new.process_and_compare_pdfs(reduced_1, reduced_2)

for difference in result1:
    print(difference)
