import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import networkx as nx
from IPython.display import display


class FinalProject:

    # init method or constructor
    def __init__(self, ucf_videos):
        self.video = []
        self.true_label = []
        self.predicted_label = []
        self.ucf_videos = ucf_videos
        self.build()

    def build(self):
        self.data_setup()
        self.dataframe_setup()
        self.setup_csv()
        self.setup_colors()
        # Visualize all predictions charts
        # self.visualize_A()
        # self.visualize_B()
        # Statistics values that derived from  Group of Actions
        # self.visualize_c()
        # self.visualize_d()
        # Classification Report
        # self.report_classification_handler()
        # Confusion Matrix Handlers
        # self.confusion_matrix_handler()
        # self.plot_confusion_matrix_handler()
        # self.plot_confusion_matrix_normalized_handler()

    def data_setup(self):
        for i in range(len(self.ucf_videos)):
            self.video.append(self.ucf_videos[i])
            self.true_label.append(self.ucf_videos[i][2:-12])

    def dataframe_setup(self):
        self.videos_df = pd.DataFrame(
            {
                "video": self.video,
                "true_label": self.true_label
            })
        display(self.videos_df)

    def setup_csv(self):
        self.data = pd.read_excel('Final_Video_V2.xlsx')
        display(self.data)
        self.groups = self.data.true_label_group.unique()
        display(self.groups)

    def setup_colors(self):
        self.colors = ['#283d3b', '#197278', '#edddd4', '#c44536', '#772e25']
        self.colors_group = dict(zip(self.groups, self.colors))
        display(self.colors_group)
        self.data['color'] = self.data.true_label_group.str.lower().map(self.colors_group)
        display(self.data)

        ## for statistacal actions
        self.groups2 = ['Human-Object Interaction', 'Body-Motion Only', 'Human-Human Interaction',
                        'Playing Musical Instruments', 'Sports']
        self.colors2 = ['#003049', '#d62828', '#f77f00', '#fcbf49', '#eae2b7']
        self.colors_group2 = dict(zip(self.groups2, self.colors2))

    def graph_bar_prob(self, true_label):
        sns.barplot(data=self.data[self.data.true_label == true_label], x='predicted_label', y='probability',
                    palette='Greens',
                    order=self.data[self.data.true_label == true_label].groupby('predicted_label')[
                        'probability'].mean().sort_values(ascending=False).index.values)

    def visualize_A(self):
        index = 0
        plt.figure(figsize=(40, 80))
        for i in range(len(self.data.true_label.unique())):
            name = self.data.true_label.unique()[index]
            index = index + 1
            plt.subplot(21, 6, index)
            G = nx.from_pandas_edgelist(self.data[self.data.true_label == name],
                                        source='true_label',
                                        target='predicted_label',
                                        edge_attr='probability'
                                        )
            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["probability"] > 0.9]
            esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["probability"] <= 0.9]
            pos = nx.spring_layout(G, seed=7)
            nx.draw_networkx_nodes(G, pos, node_size=700)
            # edges
            nx.draw_networkx_edges(G, pos, edgelist=elarge, width=8, alpha=0.8, edge_color='green')
            nx.draw_networkx_edges(
                G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="b", style="dashed"
            )
            # labels
            nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
            plt.title(name)

    def visualize_B(self):
        index = 0
        plt.figure(figsize=(40, 80))
        for i in range(len(self.data.true_label.unique())):
            name = self.data.true_label.unique()[index]
            index = index + 1
            plt.subplot(21, 6, index)
            self.graph_bar_prob(name)
            plt.title(name)
        plt.tight_layout()

    def visualize_c(self):
        self.order = self.data.groupby(['true_label_group', 'color'])[['probability']].count()
        self.order.reset_index(level=1, inplace=True)
        self.color_order2 = self.order.sort_values(by='probability', ascending=False)['color']

        plt.figure(figsize=(10, 5))
        sns.countplot(data=self.data, x='true_label_group',
                      order=self.data.true_label_group.value_counts().index,
                      palette=self.color_order2)
        plt.xticks(rotation=90)
        plt.xlabel("Groups")
        plt.ylabel("Count")
        plt.title('Number of Videos Used by Category')
        plt.show()

    def visualize_d(self):
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.data, y='probability', x='true_label_group', color='#197278',
                    order=self.data.groupby('true_label_group')['probability'].mean().sort_values(
                        ascending=False).index.values)
        plt.xticks(rotation=90)
        plt.xlabel("Groups")
        plt.ylabel("Probability")
        plt.title('Distribution by Category')
        plt.show()

    def report_classification_handler(self):
        print(classification_report(self.data.true_label_group, self.data.predicted_label_group))

    def confusion_matrix_handler(self):
        self.c_matrix = confusion_matrix(self.data.true_label_group, self.data.predicted_label_group)

    def plot_confusion_matrix_handler(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(self.c_matrix,
                                 columns=self.data.true_label_group.sort_values().unique(),
                                 index=self.data.true_label_group.sort_values().unique()), annot=True, fmt="d",
                    cmap='Greens')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title('Confusion Matrix')
        plt.show()

    def plot_confusion_matrix_normalized_handler(self):
        plt.figure(figsize=(8, 6))

        sns.heatmap(pd.DataFrame(self.c_matrix / self.c_matrix.astype(np.float).sum(axis=1),
                                 columns=self.data.true_label_group.sort_values().unique(),
                                 index=self.data.true_label_group.sort_values().unique()), annot=True,
                    cmap='Greens')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title('Normalized Confusion Matrix')
        plt.show()

    def charts_for_predictions(self):
        #chart dfown predictions 1
        self.visualize_A()
        self.visualize_B()

    def statistics_values_from_actions_grouped_up(self):
        # Statistics values that derived from  Group of Actions 2
        self.visualize_c()
        self.visualize_d()

    def classification_report(self):
        # call for classification reporting 3
        self.report_classification_handler()

    def confusion_matrix_handler_report_main(self):
        # derives conf matrix normalize and non normalized 4
        self.confusion_matrix_handler()
        self.plot_confusion_matrix_handler()
        self.plot_confusion_matrix_normalized_handler()

    def predict_video(self, sample_video):
        video_path = fetch_ucf_video(sample_video)
        sample_video = load_video(video_path)
        model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]
        logits = i3d(model_input)['default'][0]
        probabilities = tf.nn.softmax(logits)
        return probabilities


project_object = FinalProject(ucf_videos)