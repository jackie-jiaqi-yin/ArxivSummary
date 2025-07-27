import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set, Tuple
import networkx as nx
from collections import defaultdict, Counter
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from typing import Optional


class TopicSimilarityAnalyzer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.85):
        """
        Initialize the analyzer with a sentence transformer model

        Args:
            model_name: The name of the sentence transformer model to use
            similarity_threshold: Threshold for considering topics as similar (default: 0.85)
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def compute_embeddings(self, topics: List[str]) -> np.ndarray:
        """Compute embeddings for all topics"""
        return self.model.encode(topics, convert_to_tensor=True)

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between all topic pairs"""
        return cosine_similarity(embeddings)

    def find_topic_groups(self, topics: List[str]) -> Dict[str, Set[str]]:
        """
        Find groups of similar topics using graph-based clustering

        Args:
            topics: List of preprocessed topics

        Returns:
            Dictionary mapping representative topics to sets of similar topics
        """
        if not topics:
            return {}

        # Compute embeddings and similarity matrix
        embeddings = self.compute_embeddings(topics)
        similarity_matrix = self.compute_similarity_matrix(embeddings)

        # Create a graph where nodes are topics and edges represent similarity above threshold
        G = nx.Graph()
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    G.add_edge(topics[i], topics[j], weight=similarity_matrix[i, j])

        # Find connected components (topic groups)
        topic_groups = {}
        for component in nx.connected_components(G):
            # Find the most central topic in each group to use as representative
            if len(component) > 1:
                subgraph = G.subgraph(component)
                centrality = nx.degree_centrality(subgraph)
                representative = max(centrality.items(), key=lambda x: x[1])[0]
                topic_groups[representative] = set(component)
            else:
                # If component has only one topic, use it as its own representative
                topic = list(component)[0]
                topic_groups[topic] = {topic}

        return topic_groups

    def count_topic_frequency(self, topic_groups: Dict[str, Set[str]],
                              original_topics: List[str]) -> Dict[str, dict]:
        """
        Count frequency of each topic group in the original topics list

        Args:
            topic_groups: Dictionary of representative topics to sets of similar topics
            original_topics: List of original topics (can contain duplicates)

        Returns:
            Dictionary containing frequency counts and percentages for each group
        """
        # Create a mapping from each topic to its representative
        topic_to_rep = {}
        for rep, similar_topics in topic_groups.items():
            for topic in similar_topics:
                topic_to_rep[topic] = rep

        # Count occurrences of each topic in original list
        group_counts = Counter()
        for topic in original_topics:
            if topic in topic_to_rep:
                group_counts[topic_to_rep[topic]] += 1

        total_topics = len(original_topics)

        # Create detailed statistics for each group
        topic_stats = {}
        for rep, count in group_counts.items():
            topic_stats[rep] = {
                'count': count,
                'percentage': (count / total_topics) * 100 if total_topics > 0 else 0,
                'similar_topics': list(topic_groups[rep]),
                'unique_variations': len(topic_groups[rep])
            }

        return topic_stats

    def create_summary_dataframe(self, topic_stats: Dict[str, dict]) -> pd.DataFrame:
        """
        Create a pandas DataFrame with summary statistics
        """
        data = []
        for rep, stats in topic_stats.items():
            data.append({
                'representative_topic': rep,
                'count': stats['count'],
                'percentage': round(stats['percentage'], 2),
                'unique_variations': stats['unique_variations'],
                'similar_topics': ', '.join(stats['similar_topics'])
            })

        df = pd.DataFrame(data)
        # Sort by count in descending order
        df = df.sort_values('count', ascending=False)
        return df

    def get_top_topics(self, topic_stats: Dict[str, dict],
                       top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the top N most frequent topics
        """
        return sorted(
            [(rep, stats['count']) for rep, stats in topic_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

    def print_summary(self, topic_stats: Dict[str, dict]):
        """
        Print a human-readable summary of topic statistics
        """
        total_mentions = sum(stats['count'] for stats in topic_stats.values())
        total_unique = sum(stats['unique_variations'] for stats in topic_stats.values())

        print(f"Topic Analysis Summary")
        print(f"=====================")
        print(f"Total topic mentions: {total_mentions}")
        print(f"Total unique variations: {total_unique}")
        print(f"Number of topic groups: {len(topic_stats)}")
        print("\nTop 10 Topic Groups by Frequency:")
        print("--------------------------------")

        # Sort topics by count and get top 10
        sorted_topics = sorted(topic_stats.items(),
                               key=lambda x: x[1]['count'],
                               reverse=True)[:10]

        for rep, stats in sorted_topics:
            print(f"\nRepresentative Topic: {rep}")
            print(f"Count: {stats['count']} ({stats['percentage']:.1f}%)")
            print(f"Unique Variations: {stats['unique_variations']}")
            print("Similar Topics:")
            for topic in stats['similar_topics'][:5]:  # Show first 5 similar topics
                print(f"  - {topic}")
            if len(stats['similar_topics']) > 5:
                print(f"  ... and {len(stats['similar_topics']) - 5} more")

    def analyze_topics(self, topics: List[str]) -> Tuple[Dict[str, Set[str]], Dict[str, dict], pd.DataFrame]:
        """
        Complete topic analysis pipeline

        Args:
            topics: List of topics to analyze

        Returns:
            Tuple containing:
            - Topic groups dictionary
            - Topic statistics dictionary
            - Summary DataFrame
        """
        # Find topic groups
        topic_groups = self.find_topic_groups(topics)

        # Get frequency statistics
        topic_stats = self.count_topic_frequency(topic_groups, topics)

        # Create summary DataFrame
        summary_df = self.create_summary_dataframe(topic_stats)

        return topic_groups, topic_stats, summary_df


    def create_topic_bar_chart(self,
                               topic_stats: Dict[str, dict],
                               top_n: int = 20,
                               width: int = 1000,
                               height: int = 600,
                               title: str = "Topic Frequency Distribution",
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive bar chart of topic frequencies

        Args:
            topic_stats: Dictionary containing topic statistics
            top_n: Number of top topics to display (default: 20)
            width: Width of the plot in pixels (default: 1000)
            height: Height of the plot in pixels (default: 600)
            title: Title of the plot (default: "Topic Frequency Distribution")
            save_path: Optional path to save the plot as HTML (default: None)

        Returns:
            Plotly Figure object
        """
        # Sort topics by count and get top N
        sorted_topics = sorted(
            topic_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:top_n]

        # Prepare data for plotting
        topics = []
        counts = []
        percentages = []
        hover_texts = []

        for topic, stats in sorted_topics:
            # Truncate long topic names for better display
            display_topic = topic if len(topic) < 40 else topic[:37] + "..."
            topics.append(display_topic)
            counts.append(stats['count'])
            percentages.append(stats['percentage'])

            # Create hover text with similar topics
            similar_topics = stats['similar_topics'][:5]  # Show top 5 similar topics
            if len(stats['similar_topics']) > 5:
                similar_topics.append("...")

            hover_text = f"Topic: {topic}<br>" + \
                         f"Count: {stats['count']}<br>" + \
                         f"Percentage: {stats['percentage']:.1f}%<br>" + \
                         f"Unique Variations: {stats['unique_variations']}<br>" + \
                         "Similar Topics:<br>" + \
                         "<br>".join(f"- {t}" for t in similar_topics)
            hover_texts.append(hover_text)

        # Create figure
        fig = go.Figure()

        # Add bars
        fig.add_trace(
            go.Bar(
                x=counts,
                y=topics,
                orientation='h',
                text=[f"{p:.1f}%" for p in percentages],
                textposition='auto',
                hovertext=hover_texts,
                hoverinfo='text',
                marker_color='rgb(55, 83, 109)',
                marker_line_color='rgb(25, 25, 25)',
                marker_line_width=1
            )
        )

        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            },
            xaxis_title="Frequency Count",
            yaxis_title="Topics",
            width=width,
            height=height,
            yaxis={'autorange': 'reversed'},  # Show highest frequency at top
            template='plotly_white',
            hoverlabel={'bgcolor': 'white'},
            margin=dict(l=20, r=20, t=60, b=20)
        )

        # Adjust font sizes
        fig.update_yaxes(tickfont={'size': 12})
        fig.update_xaxes(tickfont={'size': 12})

        # Save if path provided
        if save_path:
            pio.write_html(fig, save_path)

        return fig

    def analyze_topics_with_viz(self,
                                topics: List[str],
                                create_plot: bool = True,
                                save_path: Optional[str] = None) -> Tuple[Dict[str, Set[str]],Dict[str, dict], pd.DataFrame, Optional[go.Figure]]:
        """
        Complete topic analysis pipeline with visualization

        Args:
            topics: List of topics to analyze
            create_plot: Whether to create visualization (default: True)
            save_path: Optional path to save the plot (default: None)

        Returns:
            Tuple containing:
            - Topic groups dictionary
            - Topic statistics dictionary
            - Summary DataFrame
            - Plotly Figure object (if create_plot=True)
        """
        # Find topic groups
        topic_groups = self.find_topic_groups(topics)

        # Get frequency statistics
        topic_stats = self.count_topic_frequency(topic_groups, topics)

        # Create summary DataFrame
        summary_df = self.create_summary_dataframe(topic_stats)

        # Create visualization if requested
        fig = None
        if create_plot:
            fig = self.create_topic_bar_chart(topic_stats, save_path=save_path)

        return topic_groups, topic_stats, summary_df, fig


