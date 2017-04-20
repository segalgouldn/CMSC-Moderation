import time  # For seeing how long things take to run.
import requests  # For downloading webpages.
import json  # For serializing files.
import re  # For cleaning up parsed outputs.
from bs4 import BeautifulSoup, Comment  # For parsing html into plain text.
from collections import OrderedDict  # For keeping good track of things (in order).
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer  # For getting useful data out of text.
from sklearn.naive_bayes import MultinomialNB  # For text classification.
from sklearn.model_selection import GridSearchCV  # For finding the best classification model parameters.
from pprint import pprint  # Just for the eyes.
from sklearn.pipeline import Pipeline  # For easy modeling.


# A Moderation project by Noah Segal-Gould which
# builds a text classification model on Bard College courselists.
class Moderation:
    def __init__(self, urls_json_filename="urls.json"):  # Take the JSON of all URLs of courselists by default.
        self.urls = json.load(fp=open(urls_json_filename), object_pairs_hook=OrderedDict)  # Keep the OrderedDict Data Structure when serialized.

        # Set a bunch of class variables for later modification.
        self.htmls = None
        self.departments = None
        self.training_set = None
        self.model = None

    # Give back the total number of courselists in the parsed file.
    def __len__(self):
        return sum([len(courselist) for semester, courselist in self.htmls.items()])

    # Load the courselists pre-parsed from a JSON.
    def load_courselists(self, htmls_json_filename="htmls.json"):
        self.htmls = json.load(fp=open(htmls_json_filename), object_pairs_hook=OrderedDict)
        return self.htmls

    # Manually download and scrape courselist HTMLs into useful courselist plaintext.
    def download_courselists(self):
        htmls = self.urls.copy()  # Make my own temporary copy of the URLs OrderedDict.
        start_time = time.time()  # Check what time it was before we did any work.
        for semester, courselists in htmls.items():
            for courselist in courselists:
                html = requests.get(courselist["url"]).text
                html_soup = BeautifulSoup(html, "html.parser")
                pretty_html = html_soup.prettify()

                # Check for HTML comments and take them out of the soup.
                for element in html_soup.find_all(string=lambda comment: isinstance(comment, Comment)):
                    element.extract()

                # Only accept <p>...</p> content from the HTML, and fix extra spaces / missing spaces.
                parsed_list = [p.get_text("|", strip=True) for p in html_soup.find_all("p")]
                parsed_string = " ".join([re.sub(r'\.([a-zA-Z])', r'. \1', p.replace("|", " ").replace("\n", " ").replace("\r", " ").replace("  ", " ").replace("   ", " ").replace("    ", " ").replace(" :", ":")) for p in parsed_list if "." in p])
                courselist.update(OrderedDict({"html": pretty_html}))
                courselist.update(OrderedDict({"parsed": parsed_string}))

                # Mostly for testing purposes: confirms values are strings.
                if not isinstance(courselist["html"], str):
                    courselist["html"] = None
        self.htmls = htmls
        self.departments = sorted(set([result["department"] for semester, results in self.htmls.items() for result in results]))
        print("Finished downloading and parsing in {} seconds".format(time.time() - start_time))  # That was useful!
        json.dump(obj=htmls, fp=open("htmls.json", "w"), indent=4)  # Save it to a file!
        return self.htmls

    # Academic departments are really inconsistent. We should probably remove certain departments from the dataset.
    # What will we keep?
    def remove_excess_departments(self, departments_to_keep, htmls_json_filename="htmls.json"):
        self.htmls = json.load(open(htmls_json_filename), object_pairs_hook=OrderedDict)
        self.departments = sorted(set([result["department"] for semester, results in self.htmls.items() for result in results]))
        departments = [department for department in self.departments if department not in departments_to_keep]
        htmls = OrderedDict()
        for semester, results in self.htmls.items():
            for result in results:
                if result["department"] not in departments:
                    try:
                        htmls[semester].append(result)
                    except:  # The nasty way of handling exceptions. Much too broad a case!
                        htmls[semester] = [result]
        self.htmls = htmls
        self.departments = sorted(set([result["department"] for semester, results in self.htmls.items() for result in results]))
        return self.htmls

    # Make format that allows us to easily feed the classifier: [("<text>", "<department>"),...]
    def build_training_set(self):
        courselists = [(result["parsed"], result["department"],) for semester, results in self.htmls.items() for result in results]
        self.training_set = courselists
        return self.training_set

    # Either set your own parameters for the classifier or run the optimizer (not by default)!
    def train_model(self, find_best_params=False):
        texts = [tup[0] for tup in self.training_set]  # Just the text
        classes = [tup[1] for tup in self.training_set]  # Just the labels

        # each variable corresponds to a part of the following scikit-learn pipeline
        parameters = {
            'vec__max_df': (0.5, 0.625, 0.75, 0.875, 1.0),
            'vec__max_features': (None, 5000, 10000, 20000),
            'vec__min_df': (1, 5, 10, 20, 50),
            'tfidf__use_idf': (True, False),
            'tfidf__sublinear_tf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
        }

        # Run it with the optimized values. Took 69.8 minutes to run.
        pipeline = Pipeline([
            ('vec', CountVectorizer(max_df=1.0, max_features=None, min_df=20, stop_words='english')),
            ('tfidf', TfidfTransformer(norm='l1', sublinear_tf=False, use_idf=True)),
            ('clf', MultinomialNB(alpha=1))
        ])
        if find_best_params:
            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)  # Use all CPU cores and show all output.
            t0 = time.time()  # What time was it at start?
            grid_search.fit(texts, classes)
            print("done in {0}s".format(time.time() - t0))
            print("Best score: {0}".format(grid_search.best_score_))  # Show the winner.
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(list(parameters.keys())):
                print("\t{0}: {1}".format(param_name, best_parameters[param_name]))
        pipeline.fit(texts, classes)
        self.model = pipeline
        return self.model  # Save the fitted model for future use.

    # Sort the prediction probabilities with their applicable classes.
    def predict(self, text):
        return OrderedDict(sorted(zip(self.model.classes_.tolist(),
                                      self.model.predict_proba([text]).tolist()[0]),
                                  key=lambda x: x[1], reverse=True))

    # Optionally run the test on any number of departments, and any body of test text. No need to download usually.
    def test(self, download=False, departments=("Computer Science", "Economics", "Mathematics", "Art History", "Spanish"), text="language"):
        print("*" * 20 + "Test Starting" + "*" * 20)
        print("\n")
        print("Testing on string: {}".format("\"" + text + "\""))
        print("\n")

        if download:
            self.download_courselists()
        else:
            self.load_courselists()

        print("Number of total courselists: {}".format(len(self)))
        self.remove_excess_departments(departments)
        self.build_training_set()
        print("Number of courselists in training set: {}".format(len(self.training_set)))

        self.train_model()
        prediction = self.predict(text)
        print("\n")
        # pprint(self.model.get_params(deep=False))  # Would have told us which model we used, but we already know.
        pprint(prediction)

        print("\n")
        print("*" * 20 + "Test Complete" + "*" * 20)

if __name__ == "__main__":
    Moderation().test()  # Surprisingly a tie between Spanish and Computer Science: "language"
    Moderation().test(text="book")  # Unsurprisingly really hard to differentiate. Results are almost normalized.
    Moderation().test(text="learning")  # Also really hard to differentiate. Results are almost normalized.
    Moderation().test(text="algorithm")  # Exactly what I would expect.
