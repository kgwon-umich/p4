#include <iostream>
#include <string>
#include <set>
#include <map>
#include <cmath>
#include "csvstream.hpp"
using namespace std;

void print_usage_error() {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
}



set<string> unique_words(const string &str) {
  istringstream source(str);
  set<string> words;
  string word;
  while (source >> word) {
    words.insert(word);
  }
  return words;
}
class Post {
  public:
    Post(const string& label_in, const string& words_in) {
        label = label_in;
        words = unique_words(words_in);
    }

    Post(const string& label_in, const set<string> words_in) {
        label = label_in;
        words = words_in;
    }

    // ASSUMES row_in CONTAINS BOTH LABEL AND CONTENT
    Post(const map<string, string>& row_in) {
        for (auto &col:row_in) {
            if (col.first == "tag") {
                label = (string) col.second;
            }
            else if (col.first == "content") {
                content = (string) col.second;
            }
        }
        words = unique_words(content);
    }

    void print_post() const {
        cout << "  label = " << label
             << ", content = " << content << endl;
    }

    string get_label() const {
        return label;
    }

    set<string> get_unique_words() const {
        return words;
    }

    bool has_word(const string& word) const {
        return words.find(word) != words.end();
    }

    string get_content_string() const {
        return content;
    }

  private:
    string label;
    set<string> words;
    string content;
};

class classifier {
  public:
    classifier() {
        classifier("sample label");
    }
    
    classifier(const string& label_in) {
        label = label_in;
        totalPosts = 1;
    }
    
    classifier(const string& label_in, int total_in) {
        label = label_in;
        totalPosts = total_in;
    }

    void updateTotalPosts(int total_in) {
        totalPosts = total_in;
    }

    void addPost(const Post* post_in) {
        posts.push_back(post_in);
        addWords(post_in->get_unique_words());
    }

    void addWords(const set<string>& words_in) {
        words.insert(words_in.begin(), words_in.end());
    }

    double get_log_prior() const {
        return log(posts.size() / (double) totalPosts);
    }

    int get_count_of(const string& word) const {
        int count = 0;
        for (auto &post:posts) {
            if (post->has_word(word)) {
                count++;
            }
        }
        return count;
    }

    void insert_log_likelihood(const string& word, double likelihood) {
        likelihoods[word] = likelihood;
    }

    bool contains_unique_word(const string& word) {
        return words.find(word) != words.end();
    }

    double get_log_likelihood(const string& word) {
        // if there exists a likelihood
        if (likelihoods.find(word) != likelihoods.end()) {
            return likelihoods.at(word);
        }
        // if there is no likelihood already calculated, calculate it
        calculate_log_likelihood(word);
        // return the likelihood now
        return likelihoods.at(word);
    }

    // REQUIRES: word is in likelihoods
    // needed for print_advanced() (const qualifier)
    double get_log_likelihood(const string& word) const {
        return likelihoods.at(word);
    }

    bool calculate_log_likelihood(const string& word) {
        int containing = get_count_of(word);
        if (containing > 0) {
            likelihoods[word] = log(containing / (double) posts.size());
            return true;
        }
        // TODO:
        // word is not in any post: resort to two edge case equations
        return false;
    }

    bool has_log_likelihood(const string& word) const {
        return likelihoods.find(word) != likelihoods.end();
    }

    // double calculate_likelihoods() {
    //     likelihoods.clear();
    //     for (auto &word:words) {
    //         likelihoods[word] = calculate_log_likelihood(word);
    //     }
    // }

    int get_size() const {
        return posts.size();
    }

    void print_basic() const {
        cout << "  " << label
             << ", " << get_size() << " examples"
             << ", log-prior = " << get_log_prior() << endl;
    }

    // REQUIRES: calculate_likelihoods() has already been called
    void print_advanced() const {
        // only iterates through unique words
        for (auto &word:words) {
            cout << "  " << label << ":" << word
                 << ", count = " << get_count_of(word)
                 << ", log-likelihood = " << get_log_likelihood(word) << endl;
        }
    }
  private:
    string label;
    set<string> words;
    map<string, double> likelihoods;
    vector<const Post*> posts;
    int totalPosts;
};

class dataSet {
  public:
    dataSet(csvstream &csvin) {
        map<string, string> row; // map of label to content
        set<string> labelsSeen;
        while (csvin >> row) {
            // create post
            Post *p = new Post(row);
            // add post
            posts.push_back(p);

            // get words as a set
            set<string> words = p->get_unique_words();
            // concatenate current post's set of words and cumulative set of words
            allWords.insert(words.begin(), words.end());

            // if we have NOT seen this label before
            if (labelsSeen.find(p->get_label()) == labelsSeen.end()) {
                // create a new classifier
                classifier c(p->get_label());
                // push classifier to classifiers vector
                // NOTE: format: pair<'label', classifier>
                classifiers[p->get_label()] = c;
                // add the label to labelsSeen
                labelsSeen.insert(p->get_label());
            }
            // now, regardless we HAVE seen this label before
            // if we just made the classifier, we only have the label but no posts
            // so we now add a post since there's guarenteed to be a classifier
            classifiers.at(p->get_label()).addPost(p);
        }
        // now update the total # of posts member of each classifier
        for (auto &c:classifiers) {
            c.second.updateTotalPosts(get_data_size());
            // c.second.calculate_likelihoods(); // !IMPORTANT
        }
        calculate_all_likelihoods();
    }

    int get_all_posts_containing(const string& word) {
        int count = 0;
        for (auto &c:classifiers) {
            count += c.second.get_count_of(word);
        }
        return count;
    }

    void calculate_log_likelihood(const string& word, const string& label) {
        // if the classifier can't calculate (if it can, it will)
        if (!classifiers.at(label).calculate_log_likelihood(word)) {
            double likelihood;
            // if word is in unique words
            if (allWords.find(word) != allWords.end()) {
                likelihood = log(get_all_posts_containing(word) / 
                                        (double) posts.size());
                classifiers.at(label).insert_log_likelihood(word, likelihood);
            }
            // if word is nowhere in any training post
            else {
                likelihood = log(1 / (double) posts.size());
                classifiers.at(label).insert_log_likelihood(word, likelihood);
            }
        }
    }

    void calculate_all_likelihoods() {
        for (auto &c:classifiers) {
            for (auto &word:allWords) {
                calculate_log_likelihood(word, c.first);
            }
        }
    }

    double get_likelihood_of_label(const Post* p, const string& label) {
        double totalProb = 0;
        // add log prior of classifier
        totalProb += classifiers.at(label).get_log_prior();
        // now add log likelihoods of each word of the post
        // iterate through every unique word of the post
        for (auto &word:p->get_unique_words()) {
            // if log likelihood doesn't exist yet, calculate it
            if (!classifiers.at(label).has_log_likelihood(word)) {
                calculate_log_likelihood(word, label);
            }
            // log likelihood is guarenteed to exist now
            totalProb += classifiers.at(label).get_log_likelihood(word);
        }
        return totalProb;
    }

    const vector<Post*> get_posts() const {
        return posts;
    }

    void print_classes_data() const {
        cout << "classes:" << endl;
        for (auto &c:classifiers) {
            c.second.print_basic();
        }
    }

    void print_classifier_data() const {
        cout << "classifier parameters:" << endl;
        for (auto &c:classifiers) {
            c.second.print_advanced();
        }
    }

    void print_training_data() const {
        for (auto &post:posts) {
            post->print_post();
        }
    }

    int get_data_size() const {
        return posts.size();
    }

    int get_vocab_size() const {
        return allWords.size();
    }

    set<string> get_all_labels() const {
        set<string> labels;
        for (auto &c:classifiers) {
            labels.insert(c.first);
        }
        return labels;
    }

    ~dataSet() {
        for (auto &post:posts) {
            delete post;
        }
    }

  private:
    map<string, classifier> classifiers;
    set<string> allWords;
    vector<Post*> posts;
};

class dataSetTester {
  public:
    dataSetTester(dataSet* ds_in, csvstream& csv_in) : 
      ds(ds_in), testSet(csv_in),
      posts(testSet.get_posts()), 
      totalCorrect(0) {
        ds->calculate_all_likelihoods();
    }

    void calculate_predictions() {
        totalCorrect = 0;
        for (auto &post:posts) {
            set<string> allLabels = ds->get_all_labels();
            string predicted = *allLabels.begin();
            double highestProb = ds->get_likelihood_of_label(post, *allLabels.begin());
            double tempProb;
            for (auto &label:allLabels) {
                tempProb = ds->get_likelihood_of_label(post, label);
                // cout << "prob of: " << label
                //      << " given post: " << post.get_content_string()
                //      << "\nis " << tempProb << endl;
                if (tempProb > highestProb) {
                    highestProb = tempProb;
                    predicted = label;
                }
            }
            if (post->get_label() == predicted) {
                totalCorrect++;
            }
            results.push_back(make_tuple(post, predicted, highestProb));
        }
    }

    void print_results() {
        cout << "test data:" << endl;
        for (auto &result:results) {
            cout << "  correct = " << get<0>(result)->get_label()
                 << ", predicted = " << get<1>(result)
                 << ", log-probability score = " << get<2>(result)
                 << endl;
            cout << "  content = " << get<0>(result)->get_content_string()
                 << endl;
            cout << endl;
        }
        cout << "performance: " << totalCorrect
             << " / " << posts.size()
             << " posts predicted correctly"
             << endl;
    }
    
  private:
    dataSet* ds;
    dataSet testSet;
    const vector<Post*> posts;
    vector<tuple<Post*, string, double>> results;
    int totalCorrect;
};


int main(int argc, char *argv[]) {
    if (argc != 2 && argc != 3) {
        print_usage_error();
        return 1;
    }

    csvstream data_in((string) argv[1]);

    // bool test = false; // determines whether the model will be tested later

    dataSet d(data_in);

    // first, configure our cout rounding
    cout.precision(3);

    if (argc == 3) {
        csvstream test_in((string) argv[2]);
        dataSetTester dst(&d, test_in);
        cout << "trained on " << d.get_data_size() << " examples" << endl;
        cout << endl;
        dst.calculate_predictions();
        dst.print_results();

    } else {        
        cout << "training data:" << endl;
        d.print_training_data();
        cout << "trained on " << d.get_data_size() << " examples" << endl;
        cout << "vocabulary size = " << d.get_vocab_size() << endl;
        cout << endl;
        d.print_classes_data();
        d.print_classifier_data();
        cout << endl;
    }
}
