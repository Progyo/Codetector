import datetime
import re
import time
from typing import Any
from codetector.dataset import HuggingFaceDataset
from codetector.samples.abstract import MappableMixin
from codetector.filters import Filter, BeforeDateFilter
from codetector.samples import CodeSample


class LeetcodePreDataset(HuggingFaceDataset):
    """
    Implementation for the Hugging Face LeetCode (pre ChatGPT) dataset.
    HF Link: https://huggingface.co/datasets/ibragim-bad/leetcode_solutions
    """

    def __init__(self,  filters : list[Filter] = [], checkPointPath = None):
        super().__init__('ibragim-bad/leetcode_solutions',
                         '3a57bcb7056bdb1a866fe95f1307a5e088691039',
                         'train', filters=filters, checkPointPath=checkPointPath)

        #https://stackoverflow.com/questions/19801727/convert-datetime-to-unix-timestamp-and-convert-it-back-in-python
        #Used this commit date: https://github.com/hendrycks/apps/commit/b064a366910559104b2b6a1f891220a45ace922e
        # self.__dsTime : int = time.mktime(datetime.date(2021,4,24).timetuple())

        #https://stackoverflow.com/questions/45279912/select-all-code-in-md-codeblock
        
        self.__regex = re.compile(r"`{3}[a-z|+]*[\r\n|\n]*(.*?)[\r\n]`{3}", re.S)


    def __dateRegression(self, problemId: int) -> int:
        """
        Given the problem id, calculate an approximate date of the problem.
        Return the unix epoch time.
        """

        #Using formula problemId * 0.7799434140207482 = 1.044161253570283 * day + 1827.357277250333
        # <=> day = (problemId * 0.7799434140207482 - 1827.357277250333)/1.044161253570283

        #Day here is the relative day to 'firstDay' = 2022-06-24

        firstDay = datetime.date(2022,6,24)

        day = (problemId * 0.7799434140207482 - 1827.357277250333)/1.044161253570283

        #https://stackoverflow.com/questions/6871016/adding-days-to-a-date-in-python
        actualDay = firstDay + datetime.timedelta(days=day)

        return int(time.mktime(actualDay.timetuple()))


    def toCodeSample(self, row: dict[str, Any]) -> CodeSample | None:
        

        #Extract code
        #https://stackoverflow.com/questions/4666973/how-to-extract-the-substring-between-two-markers
        try:
            code = self.__regex.search(row['solution']).group(1)
        except AttributeError:
            # print(json.dumps({'code': row['solution']}))
            # exit()
            return None

        prompt = row['question']

        date = self.__dateRegression(row['problem_id'])

        f = BeforeDateFilter(ymd=(2022,11,1))

                                                
        sample = CodeSample.fromLanguage(row['lang'])(content=code,
                                                prompt=prompt,
                                                originalPrompt=str(row['problem_id']),
                                                generatorTag='human',
                                                datasetTag=self.getTag(),
                                                timestamp=date,
                                                topK=None,
                                                topP=None,
                                                temperature=None)
                                      
        # print(datetime.datetime.fromtimestamp(date,datetime.timezone.utc))
        passes = f.isAllowed(sample)

        # if not passes:
        #     print(sample.getDate())

        return sample if passes else None
    

    def getTag(self):
        return 'hf_leetcode-pre'
    
    def getContentType(self) -> type[MappableMixin]:
        return CodeSample