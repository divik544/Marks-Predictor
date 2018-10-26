import numpy as np
import pandas as pd
import pymc3 as pm
from tkinter import *
from tkinter import ttk
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
np.random.seed(42)


class Window(Frame):

    def __init__(self, master):
        Frame.__init__(self, master)
        self.master = master
        self.button_font = ('Helvetica', 12)
        self.label_font = ('Consolas', 12)
        self.init_window()

    def newselection(self, event):
        global pred
        self.value_of_combo = self.box.get()
        if self.value_of_combo == 'student-por.csv':
            self.loadlabel.config(text="Loading Dataset and Training Model")
            pred = Predictor()
            self.loadlabel.config(text="Model Trained")

    def init_window(self):
        self.master.title("Grade Predictor")
        self.pack(fill=BOTH, expand=1)
        self.init_labels()
        self.init_entrys()
        self.box_value = StringVar()
        self.box = ttk.Combobox(self, textvariable=self.box_value, state='readonly')
        self.box['values'] = ('Select File', 'student-por.csv')
        self.box.current(0)
        self.box.place(x=300, y=100)
        self.box.bind("<<ComboboxSelected>>", self.newselection)
        button = Button(self, text="Predict Grade", command=button_action, bg="Cyan", fg="Black", font=self.button_font)
        button.place(x=100, y=520)
        testbut1 = Button(self, text="Load Sample 1", command=self.set_test1, bg="Cyan", fg="Black", font=self.button_font)
        testbut2 = Button(self, text="Load Sample 2", command=self.set_test2, bg="Cyan", fg="Black", font=self.button_font)
        testbut1.place(x=280, y=520)
        testbut2.place(x=470, y=520)

    def init_labels(self):
        self.loadlabel = Label(self, text="", font=self.label_font)
        self.gradelab = Label(self, text="Student with given parameters is expected to have the grade", font=self.label_font)
        self.actgradlab = Label(self, text="", font=self.label_font)
        datalabel = Label(self, text="Select Dataset", font=self.label_font)
        medulab = Label(self, text="Mother edu", font=self.label_font)
        failab = Label(self, text="Failures", font=self.label_font)
        hedulab = Label(self, text="Higher edu", font=self.label_font)
        stlab = Label(self, text="Studytime", font=self.label_font)
        fedulab = Label(self, text="Father edu", font=self.label_font)
        ablab = Label(self, text="Absences", font=self.label_font)
        desmedu = Message(self, text="0 - none, 1 - primary education, 2 - 5th to 9th grade, 3-secondary education or "
                                   "4 - higher education", font=self.label_font,width=700)
        desab = Message(self,text="From 0 to 93",font=self.label_font, width=600)
        desfail = Message(self,text="1 or 2, else 4", font=self.label_font,width=600)
        deshedu = Message(self,text="0-yes 1-no",font=self.label_font,width=600)
        desst = Message(self, text="Weekly Study Time 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, "
                                   "or 4 - >10 hours",font=self.label_font,width=600)
        desfedu = Message(self,text="Same as Medu",width=600,font=self.label_font)
        datalabel.place(x=100, y=100)
        self.loadlabel.place(x=570, y=100)
        medulab.place(x=100, y=200)
        desmedu.place(x=530, y=180)
        failab.place(x=100, y=250)
        desfail.place(x=530, y=250)
        hedulab.place(x=100, y=300)
        deshedu.place(x=530, y=300)
        stlab.place(x=100, y=350)
        desst.place(x=530, y=340)
        fedulab.place(x=100, y=400)
        desfedu.place(x=530, y=400)
        ablab.place(x=100, y=450)
        desab.place(x=530, y=450)
        self.gradelab.place(x=100, y=650)
        self.actgradlab.place(x=850, y=650)

    def init_entrys(self):
        self.tfmedu = Entry(self)
        self.tffail = Entry(self)
        self.tfhedu = Entry(self)
        self.tfst = Entry(self)
        self.tffedu = Entry(self)
        self.tfab = Entry(self)
        self.tfmedu.place(x=300, y=200)
        self.tffail.place(x=300, y=250)
        self.tfhedu.place(x=300, y=300)
        self.tfst.place(x=300, y=350)
        self.tffedu.place(x=300, y=400)
        self.tfab.place(x=300, y=450)

    def set_test1(self):
        self.tfmedu.delete(0, END)
        self.tffail.delete(0, END)
        self.tfhedu.delete(0, END)
        self.tfst.delete(0, END)
        self.tffedu.delete(0, END)
        self.tfab.delete(0, END)
        self.tfmedu.insert(0, 2)
        self.tffail.insert(0,0)
        self.tfhedu.insert(0, 1)
        self.tfst.insert(0, 1)
        self.tffedu.insert(0, 2)
        self.tfab.insert(0, 8)
        print("True Grade 12")

    def set_test2(self):
        self.tfmedu.delete(0, END)
        self.tffail.delete(0, END)
        self.tfhedu.delete(0, END)
        self.tfst.delete(0, END)
        self.tffedu.delete(0, END)
        self.tfab.delete(0, END)
        self.tfmedu.insert(0, 1)
        self.tffail.insert(0, 0)
        self.tfhedu.insert(0, 1)
        self.tfst.insert(0, 2)
        self.tffedu.insert(0, 1)
        self.tfab.insert(0, 6)
        print("True Grade 12")

    def get_medu(self):
        return int(self.tfmedu.get())

    def get_failures(self):
        return int(self.tffail.get())

    def get_hedu(self):
        return int(self.tfhedu.get())

    def get_studytime(self):
        return int(self.tfst.get())

    def get_fedu(self):
        return int(self.tffedu.get())

    def get_absences(self):
        return int(self.tfab.get())

    def update_grade(self):
        self.actgradlab.config(text=predicted_grade)

    def get_selection(self):
        return self.value_of_combo


def button_action():
    if pred is not None:
        pred.fetch_features()
        app.update_grade()


class Predictor:

    def __init__(self):
        self.process()

    def load_dataset(self):
        file = app.get_selection()
        self.data = pd.read_csv(file, delimiter=';', encoding="utf-8-sig")

    def process(self):
        self.load_dataset()
        df = self.data
        # Filter out grades that were 0
        df = df[~(df['G3'].isin([0, 1]))]
        df = df.rename(columns={'G3': 'Grade'})
        X_train, X_test, y_train, y_test = self.format_data(df)

        # Rename variables in train and teste
        X_train = X_train.rename(columns={'higher_yes': 'higher_edu',
                                          'Medu': 'mother_edu',
                                          'Fedu': 'father_edu'})

        X_test = X_test.rename(columns={'higher_yes': 'higher_edu',
                                        'Medu': 'mother_edu',
                                        'Fedu': 'father_edu'})
        # print(X_train.head())
        # Naive baseline is the median
        median_pred = X_train['Grade'].median()
        median_preds = [median_pred for _ in range(len(X_test))]
        true = X_test['Grade']

        # Display the naive baseline metrics
        mb_mae, mb_rmse = self.evaluate_predictions(median_preds, true)
        print('Median Baseline  MAE: {:.4f}'.format(mb_mae))
        print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))
        # Formula for Bayesian Linear Regression (follows R formula syntax
        formula = 'Grade ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns[1:]])

        # Context for the model
        with pm.Model() as normal_model:
            # The prior for the model parameters will be a normal distribution
            family = pm.glm.families.Normal()

            # Creating the model requires a formula and data (and optionally a family)
            pm.GLM.from_formula(formula, data=X_train, family=family)

            # Perform Markov Chain Monte Carlo sampling
            self.normal_trace = pm.sample(draws=2000, chains=2, tune=500, njobs=-1)

        # Print out the mean variable weight from the trace
        for variable in self.normal_trace.varnames:
            print('Variable: {:15} Mean weight in model: {:.4f}'.format(variable,
                                                                        np.mean(self.normal_trace[variable])))

    # Takes in a dataframe, finds the most correlated variables with the
    # grade and returns training and testing datasets
    def format_data(self, df):
        # Targets are final grade of student
        labels = df['Grade']

        # Drop the school and the grades from features
        df = df.drop(columns=['school', 'G1', 'G2'])

        # One-Hot Encoding of Categorical Variables
        df = pd.get_dummies(df)

        # Find correlations with the Grade
        most_correlated = df.corr().abs()['Grade'].sort_values(ascending=False)

        # Maintain the top 6 most correlation features with Grade
        most_correlated = most_correlated[:8]
        df = df.loc[:, most_correlated.index]
        df = df.drop(columns='higher_no')

        # Split into training/testing sets with 25% split
        X_train, X_test, y_train, y_test = train_test_split(df, labels,
                                                            test_size=0.25,
                                                            random_state=42)

        return X_train, X_test, y_train, y_test

    # Calculate mae and rmse
    def evaluate_predictions(self, predictions, true):
        mae = np.mean(abs(predictions - true))
        rmse = np.sqrt(np.mean((predictions - true) ** 2))

        return mae, rmse

    def fetch_features(self):
        observation = pd.Series({'Intercept': 1, 'mother_edu': app.get_medu(), 'failures': app.get_failures(),
                                 'higher_edu': app.get_hedu(), 'studytime': app.get_studytime(),
                                 'father_edu': app.get_fedu(), 'absences': app.get_absences()})
        self.query_model(self.normal_trace, observation)

    # Make predictions for a new data point from the model trace
    def query_model(self, trace, new_observation):
        global predicted_grade
        # Print information about the new observation
        print('New Observation')
        print(new_observation)
        # Dictionary of all sampled values for each parameter
        var_dict = {}
        for variable in trace.varnames:
            var_dict[variable] = trace[variable]

        # Standard deviation
        sd_value = var_dict['sd'].mean()

        # Results into a dataframe
        var_weights = pd.DataFrame(var_dict)

        # Align weights and new observation
        var_weights = var_weights[new_observation.index]

        # Means of variables
        var_means = var_weights.mean(axis=0)

        # Mean for observation
        mean_loc = np.dot(var_means, new_observation)

        # Distribution of estimates
        estimates = np.random.normal(loc=mean_loc, scale=sd_value,
                                     size=1000)

        # Plot the estimate distribution
        # plt.figure(figsize(8, 8))
        # sns.distplot(estimates, hist=True, kde=True, bins=19,
        #              hist_kws={'edgecolor': 'k', 'color': 'darkblue'},
        #              kde_kws={'linewidth': 4},
        #              label='Estimated Dist.')
        # # Plot the mean estimate
        # plt.vlines(x=mean_loc, ymin=0, ymax=0.2,
        #            linestyles='-', colors='orange', linewidth=2.5)
        # plt.title('Density Plot for New Observation');
        # plt.xlabel('Grade');
        # plt.ylabel('Density');

        # Estimate information
        print('Average Estimate = %0.4f' % mean_loc)
        predicted_grade = round(mean_loc, 4)

        print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),np.percentile(estimates, 95)))


root = Tk()
root.geometry("1200x800")
pred = None
app = Window(root)
root.mainloop()
