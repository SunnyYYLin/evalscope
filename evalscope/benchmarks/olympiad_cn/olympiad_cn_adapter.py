from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics.math_parser import extract_answer, math_equal, strip_answer_string
from evalscope.utils.logger import get_logger
from collections import defaultdict
from evalscope.models import ChatGenerationModelAdapter

_CITATION = """
@misc{he2024olympiadbench,
      title={OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems}, 
      author={Chaoqun He and Renjie Luo and Yuzhuo Bai and Shengding Hu and Zhen Leng Thai and Junhao Shen and Jinyi Hu and Xu Han and Yujie Huang and Yuxiang Zhang and Jie Liu and Lei Qi and Zhiyuan Liu and Maosong Sun},
      year={2024},
      eprint={2402.14008},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}"""
_REFERENCE = "https://github.com/OpenBMB/OlympiadBench/blob/main/inference/code/evaluators/evaluator.py"

chinese_answer_type_dict = {
	'Numerical': '数值',
	'Expression': '表达式',
	'Equation': '方程',
	'Interval': '区间'
}

# TODO: add support for image input
logger = get_logger()

@Benchmark.register(
    name='olympiad_cn',
    pretty_name='Olympiad Bench (Chinese)',
    model_adapter=ChatGenerationModelAdapter,
    dataset_id='lmms-lab/OlympiadBench',
    metric_list=['AveragePass@1'],
    few_shot_num=0,
    train_split=None,
    eval_split='test_cn',
    prompt_template='{query}',
    query_template='以下是中国{subfield}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”显式给出结果{unit_text}。\n{content}\n{question}'
)
class OlympiadBenchAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f'Warning: OlympiadBench only support text-only and question-answering tasks until now!')

    def load(self, **kwargs):
        data_dict = super().load(**kwargs)
        
        res_dict = defaultdict(lambda: defaultdict(list))
        for sub_name, sub_data_dict in data_dict.items():
            for split in [self.train_split, self.eval_split]:
                if split is None:
                    continue
                for sample_d in sub_data_dict[split]:
                    if len(sample_d['images']) == 0 and \
                        sample_d['final_answer'] is not None:
                        res_dict[sub_name][split].append(sample_d)

        return res_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.
        """
        format_content: dict[str, str] = {}
        if r'{subfield}' in self.query_template:
            format_content['subfield'] = input_d['subfield']

        if r'{multiple_answer_text}' in self.query_template:
            if input_d['is_multiple_answer']:
                multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
            else:
                multiple_answer_text = '\\boxed{答案}'
            format_content['multiple_answer_text'] = multiple_answer_text
        
        if r'{unit_text}' in self.query_template:    
            unit_text = ''
            if input_d['unit']:
                multiple_answer_text += '(单位)'
                unit_text = '，注意答案的单位不要放在\\boxed{}中'
            format_content['unit_text'] = unit_text
        
        if r'{answer_type_text}' in self.query_template:
            answer_type_text = self._get_answer_type_text(input_d)
            format_content['answer_type_text'] = answer_type_text
        
        if r'{context}' in self.query_template:
            format_content['context'] = str(input_d['context'])
            
        format_content['question'] = input_d['question']
        
        query = self.query_template.format(**format_content)
        full_prompt = self.prompt_template.format(
            query=query
        )
        
        return self.gen_prompt_data(full_prompt)
    
    def _get_single_answer_type_text(self, input_d: dict) -> str:
        for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
            if t in input_d['answer_type']:
                return chinese_answer_type_dict[t]
        exit('Error parsing answer type {}!'.format(input_d['answer_type']))
    
    def _get_answer_type_text(self, input_d: dict) -> str:
        if (input_d['answer_type'] is None or 'Need_human_evaluate' in input_d['answer_type']) or ('Tuple' in input_d['answer_type']):	# 'Tuple' has various meanings in different context, such as position or values of a series of variable, so it may lead to confusion to directly use 'tuple' in the prompt.
            full_answer_text = ''
        else:
            if not input_d['is_multiple_answer']:
                answer_text = self._get_single_answer_type_text(input_d)
                full_answer_text = f"T，答案类型为{answer_text}"
            else:
                if ',' not in input_d['answer_type']:	# Same answer type for all answers
                    answer_text = self._get_single_answer_type_text(input_d)
                    full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                else:
                    answer_types = input_d['answer_type'].split(',')
                    answer_types = [self._get_single_answer_type_text(input_d) for t in answer_types]
                    if len(set(answer_types)) == 1:
                        answer_text = answer_types[0]
                        full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                    else:
                        answer_text = ', '.join(answer_types)
                        full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}. '
        return full_answer_text

    def get_gold_answer(self, input_d: dict) -> str:
        if input_d['is_multiple_answer']:
            return ','.join(map(strip_answer_string, input_d['final_answer']))
        else:
            return strip_answer_string(input_d['final_answer'][0])

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.
        """
        # Note: Use same extraction method for both of checkpoint/service/custom
        result = strip_answer_string(extract_answer(result))
        return result

    def match(self, gold: str, pred: str) -> float:
        return math_equal(pred, gold)
    
if __name__ == '__main__':
    adapter = OlympiadBenchAdapter()
