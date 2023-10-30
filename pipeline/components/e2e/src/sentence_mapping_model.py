from concurrent.futures import process
from transformers import pipeline
import re
import torch

class PunctuationModel():
    def __init__(
            self,
            model:str = "oliverguhr/fullstop-punctuation-multilang-large",
            chunk_size:int = 50,
            overlap:int = 5
    ) -> None:        
        if torch.cuda.is_available():
            self.pipe = pipeline("ner",model, aggregation_strategy="none", device=0)
        else:
            self.pipe = pipeline("ner",model, aggregation_strategy="none")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def preprocess(self,text):
        #remove markers except for markers in numbers 
        text = re.sub(r"(?<!\d)[.,;:!?](?!\d)","",text) 
        #todo: match acronyms https://stackoverflow.com/questions/35076016/regex-to-match-acronyms
        text = text.split()
        return text

    def restore_punctuation(self,text):        
        result = self.predict(self.preprocess(text))
        return self.prediction_to_text(result)
        
    def overlap_chunks(self,lst, n, stride=0):
        """Yield successive n-sized chunks from lst with stride length of overlap."""
        for i in range(0, len(lst), n-stride):
                yield lst[i:i + n]

    def predict(self,words):
        if len(words) <= self.chunk_size:
            self.overlap = 0

        batches = list(self.overlap_chunks(words, self.chunk_size, self.overlap))

        # if the last batch is smaller than the overlap, 
        # we can just remove it
        if len(batches[-1]) <= self.overlap:
            batches.pop()

        tagged_words = []     
        for batch in batches:
            # use last batch completely
            if batch == batches[-1]: 
                self.overlap = 0
            text = " ".join(batch)
            result = self.pipe(text)      
            assert len(text) == result[-1]["end"], "chunk size too large, text got clipped"
                
            char_index = 0
            result_index = 0
            for word in batch[:len(batch)-self.overlap]:                
                char_index += len(word) + 1
                # if any subtoken of an word is labled as sentence end
                # we label the whole word as sentence end        
                label = "0"
                while result_index < len(result) and char_index > result[result_index]["end"] :
                    label = result[result_index]['entity']
                    score = result[result_index]['score']
                    result_index += 1                        
                tagged_words.append([word,label, score])
        
        assert len(tagged_words) == len(words)
        return tagged_words

    def prediction_to_text(self,prediction):
        result = ""
        for word, label, _ in prediction:
            result += word
            if label == "0":
                result += " "
            if label in ".,?-:":
                result += label+" "
        return result.strip()

if __name__ == "__main__":    
    model = PunctuationModel()

    text = "das , ist fies "
    # restore add missing punctuation
    result = model.restore_punctuation(text)
    print(result)

    clean_text = model.preprocess(text)
    labled_words = model.predict(clean_text)
    print(labled_words)