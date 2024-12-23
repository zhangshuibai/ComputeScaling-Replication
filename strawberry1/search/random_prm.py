import random

from prm_interface import PRM, StepScore


class RandomPRM(PRM):
    def __call__(self, steps: list[str]) -> list[StepScore]:
        """
        Args:
            steps (list[str]): A list of reasoning steps.

        Returns:
            list[StepScore]: Scored steps.
        """
        return [StepScore(step=step, score=random.uniform(0, 10)) for step in steps]
