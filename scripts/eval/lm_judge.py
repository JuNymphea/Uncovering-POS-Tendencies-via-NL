"""LM judge for concept relevance, instruction relevance and fluency (0-2 each).

The rubrics follow AxBench's judge; the prompts are in prompt_templates.py.
"""
import asyncio

import pandas as pd

from prompt_templates import (
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE,
)

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


#: The three rubrics, in the order compute_metrics reports them.
ALL_METRICS = ("concept", "instruction", "fluency")


class LMJudgeEvaluator():
    DEFAULT_RATING = 0.0
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.lm_model = kwargs.get("lm_model", None)
        self.concept_id = kwargs.get("concept_id", None)
        self.steer_dataset_type = kwargs.get("steer_dataset_type", None)
        # rubrics to ask the judge for; the aggregate is the harmonic mean over these
        self.metrics = tuple(kwargs.get("metrics", None) or ALL_METRICS)
        unknown = [m for m in self.metrics if m not in ALL_METRICS]
        if unknown:
            raise ValueError(f"unknown metric(s) {unknown}; known: {', '.join(ALL_METRICS)}")
        if "concept" not in self.metrics:
            raise ValueError("the concept rubric is the measurement; it cannot be dropped")

    def __str__(self):
        return 'LMJudgeEvaluator'

    def _get_rating_from_completion(self, completion):
        if "Rating:" in completion:
            rating_text = completion.split("Rating:")[-1].strip()
            rating_text = rating_text.split('\n')[0].strip()
            rating_text = rating_text.replace('[', '').replace(']', '')
            rating_text = rating_text.rstrip('.').strip('"').strip("'").strip("*").strip()
            rating = float(rating_text)
        else:
            logger.warning(f"Cannot find rating value: {completion}")
            rating = self.DEFAULT_RATING
        return rating

    def _get_ratings_from_completions(self, completions, min_rating=0.0, max_rating=2.0):
        ratings = []
        for completion in completions:
            try:
                # parses "Rating: [[x]]"
                rating = self._get_rating_from_completion(completion)
                if rating is not None and min_rating <= rating <= max_rating:
                    ratings.append(rating)
                else:
                    logger.warning(f"Invalid rating value: {rating}")
                    ratings.append(self.DEFAULT_RATING)
            except Exception as e:
                logger.warning(f"Failed to parse rating:\n\n{completion}\nError: {str(e)}")
                ratings.append(self.DEFAULT_RATING)
        return ratings
    
    def _get_ratings_from_prompts(self, prompts, api_name, min_rating=0.0, max_rating=2.0):
        async def process_batch():
            return await self.lm_model.chat_completions(
                f"{api_name}_{self.model_name}_LMJudgeEvaluator", prompts, batch_size=16
            )

        # asyncio.run closes its loop afterwards, so a later client.close() can fail
        # (evaluate.py ignores that)
        completions = asyncio.run(process_batch())
        return self._get_ratings_from_completions(completions, min_rating, max_rating), completions

    def _get_all_ratings_from_data(self, data, model_name):
        model_relevance_concept_prompts = []
        model_relevance_instruction_prompts = []
        model_fluency_prompts = []
        dataset_names = []
        for idx, row in data.iterrows():
            dataset_name = row["dataset_name"]
            input_concept = row["input_concept"]
            original_prompt = row["original_prompt"]
            generation = row["generation"]
            model_relevance_concept_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
                concept=input_concept,
                sentence=generation
            )]
                
            model_relevance_instruction_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE.format(
                instruction=original_prompt,
                sentence=generation
            )]
            model_fluency_prompts += [UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE.format(
                sentence=generation
            )]
            dataset_names += [dataset_name]
        def rate(prompts, api_name):
            if api_name not in self.metrics:
                return [self.DEFAULT_RATING] * len(prompts), []
            return self._get_ratings_from_prompts(prompts, api_name)

        model_relevance_concept_ratings, model_relevance_concept_completions = \
            rate(model_relevance_concept_prompts, "concept")
        model_relevance_instruction_ratings, model_relevance_instruction_completions = \
            rate(model_relevance_instruction_prompts, "instruction")
        model_fluency_ratings, model_fluency_completions = \
            rate(model_fluency_prompts, "fluency")
        return list(zip(model_relevance_concept_prompts, model_relevance_concept_ratings)), \
               list(zip(model_relevance_instruction_prompts, model_relevance_instruction_ratings)), \
               list(zip(model_fluency_prompts, model_fluency_ratings)), \
               model_relevance_concept_completions, model_relevance_instruction_completions, model_fluency_completions, dataset_names

    def compute_metrics(self, data, write_to_dir=None):
        """Rate each row 0-2 on the requested rubrics and aggregate by harmonic mean.

        The aggregate is 0 if any rating is 0. Returns means per factor plus the raw ratings.
        """
        logger.warning(
            f"Starting task for concept_id: {self.concept_id}, "
            f"model: {self.model_name}, evaluator: {self.__str__()}")
        data_copy = data.copy()
        
        model_relevance_concept_ratings, model_relevance_instruction_ratings, model_fluency_ratings, \
            model_relevance_concept_completions, model_relevance_instruction_completions, model_fluency_completions, dataset_names = \
            self._get_all_ratings_from_data(data_copy, self.model_name)
        
        all_relevance_concept_ratings = []
        all_relevance_instruction_ratings = []
        all_fluency_ratings = []
        all_aggregated_ratings = []

        for i in range(len(model_relevance_concept_ratings)):
            all_relevance_concept_ratings += [model_relevance_concept_ratings[i][-1]]
            all_relevance_instruction_ratings += [model_relevance_instruction_ratings[i][-1]]
            all_fluency_ratings += [model_fluency_ratings[i][-1]]

            def harmonic_mean(scores):
                # 0 if any rating is 0
                if 0 in scores:
                    return 0
                return len(scores) / sum(1/s for s in scores)
            concept_score = model_relevance_concept_ratings[i][-1]
            # kept from AxBench; dataset_name is a concept id here, so this never applies
            if dataset_names[i] == "AlpacaEvalSuppress":
                concept_score = 2 - concept_score
            by_metric = {
                "concept": concept_score,
                "instruction": model_relevance_instruction_ratings[i][-1],
                "fluency": model_fluency_ratings[i][-1],
            }
            # only the rubrics that were asked for; with just "concept" the
            # aggregate is that rating itself
            model_scores = [by_metric[m] for m in self.metrics]


            model_score = harmonic_mean(model_scores)
            all_aggregated_ratings += [model_score]

        metrics = {
            "metrics_judged": list(self.metrics),
            "lm_judge_rating": [],
            "relevance_concept_ratings": [],
            "relevance_instruction_ratings": [],
            "fluency_ratings": [],
            "factor": [],
            "raw_relevance_concept_ratings": all_relevance_concept_ratings,
            "raw_relevance_instruction_ratings": all_relevance_instruction_ratings,
            "raw_fluency_ratings": all_fluency_ratings,
            "raw_aggregated_ratings": all_aggregated_ratings,
            "relevance_concept_completions": model_relevance_concept_completions,
            "relevance_instruction_completions": model_relevance_instruction_completions,
            "fluency_completions": model_fluency_completions
        }
        data_copy[f"{self.model_name}_lm_judge_rating"] = all_aggregated_ratings
        data_copy[f"{self.model_name}_relevance_concept_ratings"] = all_relevance_concept_ratings
        data_copy[f"{self.model_name}_relevance_instruction_ratings"] = all_relevance_instruction_ratings
        data_copy[f"{self.model_name}_fluency_ratings"] = all_fluency_ratings

        # group by factor only and compute means
        grouped = data_copy.groupby("factor")
        for factor, group in grouped:
            metrics["lm_judge_rating"].append(group[f"{self.model_name}_lm_judge_rating"].mean())
            metrics["relevance_concept_ratings"].append(group[f"{self.model_name}_relevance_concept_ratings"].mean())
            metrics["relevance_instruction_ratings"].append(group[f"{self.model_name}_relevance_instruction_ratings"].mean())
            metrics["fluency_ratings"].append(group[f"{self.model_name}_fluency_ratings"].mean())
            metrics["factor"].append(factor)

        return metrics