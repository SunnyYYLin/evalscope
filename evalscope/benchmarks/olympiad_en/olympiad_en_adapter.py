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

english_answer_type_dict = {
	'Numerical': 'a numerical value',
	'Expression': 'an expression',
	'Equation': 'an equation',
	'Interval': 'an interval'
}

# TODO: add support for image input
logger = get_logger()

@Benchmark.register(
    name='olympiad_en',
    pretty_name='Olympiad Bench (English)',
    model_adapter=ChatGenerationModelAdapter,
    dataset_id='lmms-lab/OlympiadBench',
    metric_list=['AveragePass@1'],
    few_shot_num=0,
    train_split=None,
    eval_split='test_en',
    prompt_template='The following is an open-ended problem from an International {subfield} competition. {question}\n{answer_type_text}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is {multiple_answer_text}." and give the result explicitly{unit_text}.'
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
        subject_content = input_d['subfield']
        if input_d['is_multiple_answer']:
            multiple_answer_text = '\\boxed{multiple answers connected with commas}'
        else:
            multiple_answer_text = '\\boxed{answer}'
        unit_text = ''
        if input_d['unit']:
            multiple_answer_text += '(unit)'
            unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
        answer_type_text = self._get_answer_type_text(input_d)
        question = str(input_d['context']) + '\n' + input_d['question']
        
        full_prompt = self.prompt_template.format(
            subfield=subject_content,
            question=question,
            answer_type_text=answer_type_text,
            multiple_answer_text=multiple_answer_text,
            unit_text=unit_text
        )
        
        return self.gen_prompt_data(full_prompt)
    
    def _get_single_answer_type_text(self, input_d: dict) -> str:
        for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
            if t in input_d['answer_type']:
                return english_answer_type_dict[t]
        exit('Error parsing answer type {}!'.format(input_d['answer_type']))
    
    def _get_answer_type_text(self, input_d: dict) -> str:
        if ('Need_human_evaluate' in input_d['answer_type']) or ('Tuple' in input_d['answer_type'] or input_d['answer_type'] is None):	# 'Tuple' has various meanings in different context, such as position or values of a series of variable, so it may lead to confusion to directly use 'tuple' in the prompt.
            full_answer_text = ''
        else:
            if not input_d['is_multiple_answer']:
                answer_text = self._get_single_answer_type_text(input_d)
                full_answer_text = f"The answer of The problem should be {answer_text}. "
            else:
                if ',' not in input_d['answer_type']:	# Same answer type for all answers
                    answer_text = self._get_single_answer_type_text(input_d)
                    full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
                else:
                    answer_types = input_d['answer_type'].split(',')
                    answer_types = [self._get_single_answer_type_text(input_d) for t in answer_types]
                    if len(set(answer_types)) == 1:
                        answer_text = answer_types[0]
                        full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
                    else:
                        answer_text = ', '.join(answer_types)
                        full_answer_text = f'The problem has multiple answers, with the answers in order being {answer_text}. '
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
        # return float(gold == pred)
        return float(math_equal(pred, gold))
    
if __name__ == '__main__':
    adapter = OlympiadBenchAdapter()
