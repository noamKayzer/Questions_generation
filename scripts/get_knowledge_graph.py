#!pip install transformers wikipedia newspaper3k GoogleNews pyvis
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
import wikipedia
from newspaper import Article, ArticleException
from GoogleNews import GoogleNews
import IPython
try:
    from pyvis.network import Network
except:
    from pyvis.network import Network
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations
    
def from_text_to_kb(text, article_url, span_length=128, article_title=None,
                    article_publish_date=None, verbose=False):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) / 
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    # create kb
    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                article_url: {
                    "spans": [spans_boundaries[current_span_index]]
                }
            }
            kb.add_relation(relation, article_title, article_publish_date)
        i += 1

    return kb

def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article

def from_url_to_kb(url):
    article = get_article(url)
    config = {
        "article_title": article.title,
        "article_publish_date": article.publish_date
    }
    kb = from_text_to_kb(article.text, article.url, **config)
    return kb

def save_network_html(kb, filename="network.html"):
    # create network
    net = Network(directed=True, width="700px", height="700px", bgcolor="#eeeeee")

    # nodes
    color_entity = "#00FF00"
    for e in kb.entities:
        net.add_node(e, shape="circle", color=color_entity)

    # edges
    for r in kb.relations:
        net.add_edge(r["head"], r["tail"],
                    title=r["type"], label=r["type"])
        
    # save network
    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')
    net.show(filename)
import pickle

def save_kb(kb, filename):
    with open(filename, "wb") as f:
        pickle.dump(kb, f)

def load_kb(filename):
    res = None
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res

class KB():
    def __init__(self):
        self.entities = {} # { entity_title: {...} }
        self.relations = [] # [ head: entity_title, type: ..., tail: entity_title,
          # meta: { article_url: { spans: [...] } } ]
        self.sources = {} # { article_url: {...} }

    def merge_with_kb(self, kb2):
        for r in kb2.relations:
            article_url = list(r["meta"].keys())[0]
            source_data = kb2.sources[article_url]
            self.add_relation(r, source_data["article_title"],
                              source_data["article_publish_date"])

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r2):
        r1 = [r for r in self.relations
              if self.are_relations_equal(r2, r)][0]

        # if different article
        article_url = list(r2["meta"].keys())[0]
        if article_url not in r1["meta"]:
            r1["meta"][article_url] = r2["meta"][article_url]

        # if existing article
        else:
            spans_to_add = [span for span in r2["meta"][article_url]["spans"]
                            if span not in r1["meta"][article_url]["spans"]]
            r1["meta"][article_url]["spans"] += spans_to_add

    def get_wikipedia_data(self, candidate_entity):
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary
            }
            return entity_data
        except:
            return None

    def add_entity(self, e):
        self.entities[e["title"]] = {k:v for k,v in e.items() if k != "title"}

    def add_relation(self, r, article_title, article_publish_date):
        # check on wikipedia
        candidate_entities = [r["head"], r["tail"]]
        entities = [self.get_wikipedia_data(ent) for ent in candidate_entities]

        # if one entity does not exist, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        r["head"] = entities[0]["title"]
        r["tail"] = entities[1]["title"]

        # add source if not in kb
        article_url = list(r["meta"].keys())[0]
        if article_url not in self.sources:
            self.sources[article_url] = {
                "article_title": article_title,
                "article_publish_date": article_publish_date
            }

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def print(self):
        print("Entities:")
        for e in self.entities.items():
            print(f"  {e}")
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")
        print("Sources:")
        for s in self.sources.items():
            print(f"  {s}")


if __name__=='__main__':
    text = [" The crisis began with bubbles in the stock market, housing market, and also in the commodities market. It's a financial crisis that's bigger than any since the Great Depression of the 1930's. There's many different ways of thinking about a crisis like this. And I wanted to focus on one way that people think about it in terms of probability models. So, that's not the only way, it's not necessarily my favorite way. Excuse my cold. I didn't bring any water. I hope I make it through this lecture.  There was a pre-break around 2000 when the stock market collapsed around the world. But then they came back again after 2003 and they were on another boom, like a roller coaster ride. That's the narrative story. And then, what happened is, we see a bunch of institutional collapses. We saw bank failures in the U.S. and then, we saw international cooperation to prevent this from spreading like a disease. So, we had governments all over the world bailing out their banks and other companies. That's what financial theorists will think about is that actually it's not just those few big events. It's the accumulation of a lot of little events.  I'm going to talk today about probability, variance, and covariance, and regression, and idiosyncratic risk, and systematic risk. But I'm also going to, in the context of the crisis, emphasize in this lecture, breakdowns of some of the most popular assumptions that underlie financial theory. And I'm thinking particularly of two breakdowns. One is the failure of independence. And another one is a tendency for outliers or fat-tailed distributions.", " The word probability in its present meaning wasn't even coined until the 1600's. We do it by dealing with all of these little incremental shocks that affect our lives in a mathematical way. We have mathematical laws of how they accumulate. And once we understand those laws, we can we can build mathematical models of the outcomes. And then we can ask whether we should be surprised by the financial events that we've seen. It's a little bit like science, real hard science. So, for example, weather forecasters build models that are built on the theory of fluid dynamics.  People who are steeped in this tradition in finance think that what we're doing is very much like what we do when we do financial forecasts. We have a statistical model, we see all of the shocks coming in, and of course there will be hurricanes. And we can only forecast them -- you know there's a limit to how far out we can forecast them. Weather forecasters can't do that. Same thing with financial crises. We understand the probability laws, there's only a certain time horizon before which we can.", " In finance, the basic, the most basic concept that -- in finance -- is that when you invest in something, you have to do it for a time interval. And so, what is your return to investing in something? It's the increase in the price. That's p t plus 1, minus p t. Returns can be positive or negative. They can never be more than -- never be less than minus 100%. In a limited liability economy that we live in, the law says that you cannot lose more than the money you put in.  This is the mathematical expectation of a random variable x, which could be the return, or the gross return, but we're going to substitute something else. The expectation of x is the weighted sum of all possible values of x weighted by their probabilities. And the probabilities have to sum to 1. They're positive numbers, or zero, reflecting the likelihood of that random variable occurring, of that value of the random variable. This is for a discrete random variable that takes on only a finite, only a countable number of values. Gross return is always positive. It's between zero and infinity.  If you have n observations on a random variable x, you can take the sum of the x observations, summation over i equals 1 to n, and then divide that by n. That's called the average, or the mean, or sample mean, when you have a sample of n observations, which is an estimate of the expected value of x. This is called the mean or average, which you've learned long ago, OK. So, for example, if we're evaluating an investor who has invested money, you could get n observations and take an average of them.  The geometric mean makes sense only when all the x's are non-negative. If you put in a negative value, you might get a negative product, and then, if you took the nth root of that, it could be an imaginary number, so let's forget that. We're not going to apply this formula if there are any negative numbers. But it's often used, and I recommend its use, in evaluating investments. Because if you use gross return, it gives a better measure of the outcome of the investments.  If there's ever a year in which the return is minus 100%, then the geometric mean is 0. That's a good discipline. This obviously doesn't make sense as a way to evaluate investment success. But we care about more than just about central tendency when evaluating risk. We have to do other things as well, including the geometric return, variance, variance and variance. And so, you want to talk about risk, this is very fundamental to finance. What could be more fundamental than risk for finance?  If x tends to be plus or minus 1% from the mean return, the variance would probably be 1. The standard deviation is the square root of the variance. Covariance is a measure of how two different random variables move together. When IBM goes up, does General Motors go up or not? We're getting through these concepts, but I'm not going to get into these ideas here, so I'm just trying to be very basic and simple here, but they're very basic.  A measure of the co-movement of the two would be to take deviation of x from its mean times the deviation of y from it's mean, and take the average product of those. It's a positive number if, when x is high relative to its mean, y is. And it's a negative number if they tend to go in opposite directions. If GM tends to do well when IBM does poorly, then we have a negative covariance. And this is the core concept that I was talking about. Some idea of unrelatedness underlies a lot of our thinking in risk.  If two variables have a plus 1 correlation, that means they move exactly together. If they are independent, then their correlation should be zero. That's true if the random variables are independent of each other. But we're going to see that breakdown of independence is the story of this lecture. We want to think about independence as mattering a lot. And it's a model, or a core idea, but when do we know that things are independent?", " The crisis that we've seen here in the stock market is the accumulation of -- you see all these ups and downs. There were relatively more downs in the period from 2000 and 2002. But how do we understand the cumulative effect of it, which is what matters? So, we have to have some kind of probability Model. And that is a core question that made it so difficult for us to understand how to deal with such a crisis, and why so many people got in trouble dealing with this crisis.  After the 1987 crash, companies started to compute a measure of the risk to their company, called Value at Risk. Many companies had calculated numbers like this, and told their investors, we can't do too badly because there's no way that we could lose. But they were implicitly making assumptions about independence, or at least relative independence. And so, you need a probability Model to make these calculations, which is based on probability theory in order to do that. And it's not one that is easy to be precise about. It's a core concept in finance.  Companies all over the world were estimating very small numbers here, relative to what actually happened. The law of large numbers says that if I have a lot of independent shocks, and average them out, on average there's not going to be much uncertainty. It says that the variance of the average of n random variables that are all independent and identically distributed goes to 0 as the number of elements in the average goes to infinity. And so, that's a fundamental concept that underlies both finance and insurance.  The law of large numbers has to do with the idea that if I have a large number of random variables, what is the variance of -- the square root of the variance. If they're all independent, then all of the covariances are 0. So, as n goes large, you can see that the standard deviation of the mean goes to 0. The mean is divided by n. The standard deviation is equal to the squareroot of n times the squared root of one of the variables.  There's a new idea coming up now, after this recent crisis, and it's called CoVaR. It's a concept emphasized by Professor Brunnermeier at Princeton and some of his colleagues, that we have to change analysis of variance to recognize that portfolios can sometimes co-vary more than we thought. In the present environment, I think, we recognize the need for that.", ' The stock market lost something like almost half of its value between 2000 and 2002. But when I put Apple on the same plot, the computer had to, because Apple did such amazing things, it had to compress. Apple computer is the one of the breakout cases of dramatic success in investing. It went up 25 times. This incidentally is the adjusted price for Apple, because in 2005 Apple did a 2-for-1 split. You know what that means? You can see this. You might be surprised to say, wait a minute, did I hear you right?  A lot of companies, when the price hits $60 or something like that, they say, well let\'s just split all the shares in two. An investment in Apple went up 25 times, whereas an investment in the S & P 500 went up only -- well, it didn\'t go up, actually, it\'s down. Now, this is a plot showing the monthly returns on Apple. You can\'t tell from this plot that Apple went. up 25-fold. That matters a lot to an investor.  Buy Apple and your money will go up 25-fold, says Warren Buffett. Buffett: "It wasn\'t an even ride. It\'s a scary ride" Buffett: Buy Apple in one month, you lose 30% in another month, but you can\'t tell what\'s driving it up and down. He says the ride, as you\'re observing this happen, every month it goes opposite. I just goes big swings. Buffett: The ride is not so obvious because it\'s a rollercoaster ride. You can\'t see it happening unless you look at your portfolio.  In 1979, the Yale class of 1954 had a 25th reunion, and asked an investor to take a risky portfolio investment for Yale and let\'s give it to Yale on our 50th anniversary, all right? So, they got a portfolio manager, his name was Joe McNay, and they said -- they put together -- it was $375,000. It\'s like one house, you know, for all the whole class, no big deal. So, McNay decided to invest in Home Depot, Walmart, and internet stocks. On their 50th reunion in 2004, they presented Yale University with $90 million dollars.  He started liquidating in 2000, right the peak of the market. So, it must be partly luck. No one could have known that Walmart was going to be such a success. For every one of the great men and women of history, there\'s 1,000 of them that got squashed. And I think that history is like that. The people you read about in history are often just phenomenal risk takers like Joe McNey. But maybe they\'re just lucky, maybe they are just lucky.  Apple lost about a third of its value in one month in 2008. The company\'s founder Steve Jobs had pancreatic cancer in 2004, but the doctors said it\'s curable, no problem, so the stock didn\'t do anything. So, it quickly rebounded because he wasn\'t, and the company\'s stock went up because he was not cancer-stricken. Bob Greene: Maybe it\'s all those poor, all those ordinary people, living the little house, the $400,000 house, they don\'t risk it. Maybe they\'re the smart ones.  Aaron Carroll: I\'ve just told you about one blip here, but they were so many of these blips on the way, and they all have some story about the success of some Apple product, or people aren\'t buying some product. Each point represents one of the points that we saw on the market, Carroll says. Carroll: "It looks totally different, and it shows such complexity that I can\'t tell a simple narrative. Every month looks different.  The best success was in December, January of 2001, where the stock price went up 50% in one month. The reason why it looks kind of compressed on this way is, because the stock market doesn\'t move as much as Apple. The return for a stock, for the i-th stock, is equal to the market return, which is represented here by the S & P 500, plus idiosyncratic return. The variance of the stock returns is the variance -- the variance of a stock return is the sum of the market.  Apple shows a magnified response to the stock market. It goes up and down approximately one and a half times as much as stock market does on any day. Apple has a lot of idiosyncratic risk, but the aggregate economy matters, right? If you think that maybe because Apple is kind of a vulnerable company, that if the economy tanks, Apple will tank even more than the economy. If the market goes up, then it\'s even better news for Apple, even though it\'s a volatile, dangerous strategy company.  He founded Apple and Apple prospered, then had a falling out with the management, and got kind of kicked out of his own company. And then he founded Next Computer. But meanwhile, Apple started to really tank, and they finally realized they needed Steve Jobs, so they brought him back. And it turned out to be the same month that\'s the Lehman Brothers collapse occurred. This line, I thought it would have an even higher beta, but I think it\'s this point which is bringing the beta down.', ' A lot of probability theory works on the assumption that variables are normally distributed. But random variables have a habit of not behaving that way, especially in finance it seems. Benoit Mandelbrot was the discoverer of this concept, and I think the most important figure in it. Pierre Paul Levy invented the concept, as discussed in the next lecture in this week\'s Lecture on the idea of the \'normal\' distribution of random shocks to the financial economy. The bell-shaped curve is thought to be a parabola, a mathematical function.  In nature the normal distribution is not the only distribution that occurs, and that especially in certain kinds of circumstances we have more fat-tailed distributions. The way you find out that they\'re not the same, is that in extremely rare circumstances there\'ll be a sudden major jump in the variable that you might have thought couldn\'t happen. Whether it\'s Cauchy or normal, they look about the same; they look pretty much the same. But the pink line has tremendously large probability of being far out. These are the tails of the distribution.  Stock market went up 12.53% on October 30, 1929. That\'s the biggest one-day increase in the history of the stock market. But there were maybe like 20 days, I can\'t read off the chart when it did this since 1928. You can go through ten years on Wall Street and never see a drop of that magnitude. So, eventually you get kind of assured. It can\'t happen. What about an 8% drop? Well, I look at this, I say, I\'ve never seen that. It just doesn\'t happen.  The stock market crash of 1929 had two consecutive days. It went down about 12% on October 28, and then the next day it did it again. That\'s way off the charts, and if you compute the normal distribution, what\'s the probability of that? If it\'s a normal distribution and it fits the central portion, it would say it\'s virtually zero. It couldn\'t happen. Anyone have any idea what happened on October 30, 1929? It\'s obvious to me, but it\'s not obvious to you.  If you believe in normality, October 19, 1987 couldn\'t happen, Bob Greene says. He says a student raised his hand and said the stock market is "totally falling apart" Greene: The probability of a decline that\'s that negative? It\'s 10 to the minus 71 power. 1 over 10 power. That\'s an awfully small number. But there it is. It happened, Greene says, and in fact, I\'ve been teaching this course for 25 years. It just came as a complete surprise to me. I went downtown to Merrill Lynch.  The two themes are that independence leads to the law of large numbers, and it leads to some sort of stability. But that\'s not what happened in this crisis and that\'s the big question. You get big incredible shocks that you thought couldn\'t happen, and they just come up with a certain low probability.']
    #text = "Napoleon Bonaparte (born Napoleone di Buonaparte; 15 August 1769 – 5 May 1821), and later known by his regnal name Napoleon I, was a French military and political leader who rose to prominence during the French Revolution and led several successful campaigns during the Revolutionary Wars. He was the de facto leader of the French Republic as First Consul from 1799 to 1804. As Napoleon I, he was Emperor of the French from 1804 until 1814 and again in 1815. Napoleon's political and cultural legacy has endured, and he has been one of the most celebrated and controversial leaders in world history."

    text ='Jordan was founded in 1960. When Israel was established? few years after 1948'

    if isinstance(text,list):
        kb = from_text_to_kb(" ".join(text), "", verbose=True)
    else:
        kb = from_text_to_kb(text, "",span_length=128, verbose=True)
    kb.print()
    filename = "israel.html"
    save_network_html(kb, filename=filename)
    save_kb(kb, filename.split(".")[0] + ".p")
    #IPython.display.HTML(filename=filename)