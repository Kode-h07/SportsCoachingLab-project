import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import NearestNeighbors
import warnings
import sys
import io
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium
import networkx as nx
import matplotlib.font_manager as fm
from matplotlib import rcParams

# Set up Korean font explicitly
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path).get_name()
rcParams["font.family"] = font_prop

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

warnings.filterwarnings("ignore")


class SeoulDistrictAnalyzer:
    def __init__(self):
        self.gu_data = None
        self.survey_data = None
        self.combined_data = None
        self.final_scores = None

    def load_data(self):
        """Load the CSV files"""
        try:
            self.gu_data = pd.read_csv("gu_feature.csv")
            self.survey_data = pd.read_csv("preprocessed.csv")
            print("Data loaded successfully!")
            print(f"GU Features shape: {self.gu_data.shape}")
            print(f"Survey data shape: {self.survey_data.shape}")
            return True
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print(
                "Please ensure 'gu_feature.csv' and 'preprocessed.csv' are in the current directory"
            )
            return False

    def preprocess_survey_data(self):
        """Extract district information from survey data"""
        if self.survey_data is None:
            return

        # Extract district (구) from homeaddress
        def extract_gu(address):
            if pd.isna(address):
                return None
            # Look for district patterns
            districts = [
                "강남구",
                "강동구",
                "강북구",
                "강서구",
                "관악구",
                "광진구",
                "구로구",
                "금천구",
                "노원구",
                "도봉구",
                "동대문구",
                "동작구",
                "마포구",
                "서대문구",
                "서초구",
                "성동구",
                "성북구",
                "송파구",
                "양천구",
                "영등포구",
                "용산구",
                "은평구",
                "종로구",
                "중구",
                "중랑구",
            ]

            for district in districts:
                if district in address:
                    return district
            return None

        self.survey_data["home_gu"] = self.survey_data["homeaddress"].apply(extract_gu)

        # Extract closest club district
        def extract_club_gu(club_address):
            if pd.isna(club_address):
                return None
            if "성동구" in club_address:
                return "성동구"
            elif "중구" in club_address:
                return "중구"
            elif "서초구" in club_address:
                return "서초구"
            return None

        self.survey_data["closest_club_gu"] = self.survey_data["closest_club"].apply(
            extract_club_gu
        )

        print("Survey data preprocessed!")

    def calculate_demand_metrics(self):
        """Calculate demand metrics from survey data"""
        # Group by home district
        demand_by_gu = (
            self.survey_data.groupby("home_gu")
            .agg(
                {
                    "is_woman": "mean",
                    "age": "mean",
                    "office_worker": "mean",
                    "travel_time_min": "mean",
                    "homeaddress": "count",  # Number of members from each gu
                }
            )
            .rename(columns={"homeaddress": "member_count"})
        )

        # Calculate accessibility score (inverse of travel time)
        demand_by_gu["accessibility_score"] = 1 / (demand_by_gu["travel_time_min"] + 1)

        # Group by closest club to understand service demand
        club_demand = (
            self.survey_data.groupby("closest_club_gu")
            .agg({"homeaddress": "count", "travel_time_min": "mean"})
            .rename(columns={"homeaddress": "served_members"})
        )

        return demand_by_gu, club_demand

    def create_combined_features(self):
        """Combine demographic and demand features"""
        demand_by_gu, club_demand = self.calculate_demand_metrics()

        # Merge with GU demographic data
        self.combined_data = self.gu_data.merge(
            demand_by_gu, left_on="gu", right_index=True, how="left"
        )

        # Fill missing values with median/mode
        self.combined_data["member_count"] = self.combined_data["member_count"].fillna(
            0
        )
        self.combined_data["accessibility_score"] = self.combined_data[
            "accessibility_score"
        ].fillna(self.combined_data["accessibility_score"].median())
        self.combined_data["travel_time_min"] = self.combined_data[
            "travel_time_min"
        ].fillna(self.combined_data["travel_time_min"].median())

        print("Combined features created!")
        print(f"Combined data shape: {self.combined_data.shape}")

    def apply_semi_supervised_learning(self):
        """Apply semi-supervised learning to generate the final scoring"""
        # Prepare features for ML
        features = [
            "office_worker_pct",
            "women_pct",
            "20to39_pct",
            "member_count",
            "accessibility_score",
        ]
        X = self.combined_data[features].fillna(self.combined_data[features].mean())

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create semi-supervised labels based on existing data
        y = np.full(len(X), -1)  # -1 for unlabeled

        # Label districts with high member count as positive (high potential)
        high_member_threshold = self.combined_data["member_count"].quantile(0.75)
        high_member_mask = self.combined_data["member_count"] >= high_member_threshold
        y[high_member_mask] = 1

        # Label districts with very low member count as negative (low potential)
        low_member_threshold = self.combined_data["member_count"].quantile(0.25)
        low_member_mask = self.combined_data["member_count"] <= low_member_threshold
        y[low_member_mask] = 0

        # Apply Label Propagation
        label_prop = LabelPropagation(kernel="knn", n_neighbors=7)
        label_prop.fit(X_scaled, y)

        # Get predictions and probabilities
        y_pred = label_prop.predict(X_scaled)
        y_proba = label_prop.predict_proba(X_scaled)

        # Store results
        self.combined_data["ssl_prediction"] = y_pred
        self.combined_data["ssl_confidence"] = np.max(y_proba, axis=1)

        # PURE SSL SCORING: Use probability of being class 1 (high potential) as the final score
        # This is the ML model's assessment of each district's potential
        self.combined_data["final_score"] = y_proba[:, 1]  # Probability of class 1

        print("Pure Semi-Supervised Learning scoring applied!")
        print(f"Label distribution: {np.bincount(y_pred)}")
        print(
            f"Score range: {self.combined_data['final_score'].min():.3f} - {self.combined_data['final_score'].max():.3f}"
        )

        return X_scaled, y_pred, y_proba

    def calculate_comprehensive_scores(self):
        """This method is now simplified - SSL already calculated the final scores"""
        # Sort by the SSL-generated final score
        self.final_scores = self.combined_data.sort_values(
            "final_score", ascending=False
        )

        # Add some interpretable component scores for analysis purposes only
        scaler = MinMaxScaler()

        # These are for analysis/interpretation, NOT for final scoring
        demographic_features = ["office_worker_pct", "women_pct", "20to39_pct"]
        demand_features = ["member_count", "accessibility_score"]

        demo_scores = scaler.fit_transform(self.combined_data[demographic_features])
        self.combined_data["demographic_score"] = np.mean(demo_scores, axis=1)

        demand_scores = scaler.fit_transform(self.combined_data[demand_features])
        self.combined_data["demand_score"] = np.mean(demand_scores, axis=1)

        # Update final_scores to include these interpretive scores
        self.final_scores = self.combined_data.sort_values(
            "final_score", ascending=False
        )

        print("Final ranking completed using pure SSL scores!")

        # Show how SSL learned to weight different factors
        self.analyze_ssl_feature_importance()

    def analyze_ssl_feature_importance(self):
        """Analyze what the SSL model learned about feature importance"""
        print("\nSSL MODEL LEARNED PATTERNS:")
        print("=" * 40)

        # Compare features between high-score and low-score districts
        high_score = self.final_scores.head(10)  # Top 10
        low_score = self.final_scores.tail(10)  # Bottom 10

        features = [
            "office_worker_pct",
            "women_pct",
            "20to39_pct",
            "member_count",
            "accessibility_score",
        ]

        print("Feature comparison (High potential vs Low potential districts):")
        print("-" * 60)
        for feature in features:
            high_mean = high_score[feature].mean()
            low_mean = low_score[feature].mean()
            difference = high_mean - low_mean
            print(
                f"{feature:20}: High={high_mean:6.2f}, Low={low_mean:6.2f}, Diff={difference:+6.2f}"
            )

        # Show correlation of each feature with final SSL score
        print("\nFeature correlations with SSL final score:")
        print("-" * 40)
        for feature in features:
            corr = np.corrcoef(
                self.combined_data[feature], self.combined_data["final_score"]
            )[0, 1]
            print(f"{feature:20}: {corr:6.3f}")

    def generate_recommendations(self):
        """Generate recommendations using pure SSL results"""
        excluded_districts = ["서초구", "중구", "성동구"]

        # Filter out districts with existing clubs
        filtered_scores = self.final_scores[
            ~self.final_scores["gu"].isin(excluded_districts)
        ]

        print("\n" + "=" * 80)
        print("SEOUL DISTRICT ANALYSIS - PURE SSL RECOMMENDATIONS")
        print("=" * 80)
        print("(Scores generated entirely by Label Propagation ML model)")

        top_5 = filtered_scores.head(5)
        print(f"\nTOP 5 RECOMMENDED DISTRICTS (by SSL probability):")
        print("-" * 60)

        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            ssl_class = (
                "HIGH POTENTIAL" if row["ssl_prediction"] == 1 else "LOW POTENTIAL"
            )
            print(f"{i}. {row['gu']}")
            print(f"   SSL Score (P(High Potential)): {row['final_score']:.3f}")
            print(f"   SSL Classification: {ssl_class}")
            print(f"   SSL Confidence: {row['ssl_confidence']:.3f}")
            print(
                f"   Demographics: Office {row['office_worker_pct']:.1f}%, Women {row['women_pct']:.1f}%, Age 20-39 {row['20to39_pct']:.1f}%"
            )
            print(f"   Current Members: {row['member_count']:.0f}")
            if not pd.isna(row["travel_time_min"]):
                print(f"   Avg Travel Time: {row['travel_time_min']:.1f} minutes")
            print()

        print("\nSSL MODEL INSIGHTS:")
        print("-" * 30)

        # Districts classified as high potential by SSL
        high_potential = self.final_scores[self.final_scores["ssl_prediction"] == 1]
        low_potential = self.final_scores[self.final_scores["ssl_prediction"] == 0]

        print(f"Districts classified as HIGH potential: {len(high_potential)}")
        print(f"Districts classified as LOW potential: {len(low_potential)}")

        print(f"\nTop SSL-identified high-potential districts:")
        high_potential_sorted = high_potential.sort_values(
            "final_score", ascending=False
        )
        for i, (idx, row) in enumerate(high_potential_sorted.head(5).iterrows(), 1):
            print(f"  {i}. {row['gu']} (SSL score: {row['final_score']:.3f})")

        print(f"\nSSL model's most confident predictions:")
        most_confident = self.final_scores.nlargest(5, "ssl_confidence")
        for i, (idx, row) in enumerate(most_confident.iterrows(), 1):
            pred_text = "HIGH" if row["ssl_prediction"] == 1 else "LOW"
            print(
                f"  {i}. {row['gu']}: {pred_text} potential (confidence: {row['ssl_confidence']:.3f})"
            )

    def run_complete_analysis(self):
        """Run the complete analysis pipeline with pure SSL scoring"""
        print("Starting Seoul District Analysis with Pure SSL Scoring...")

        if not self.load_data():
            return

        self.preprocess_survey_data()
        self.create_combined_features()

        # SSL does both prediction AND final scoring
        X_scaled, y_pred, y_proba = self.apply_semi_supervised_learning()

        # This now just organizes results, doesn't change scores
        self.calculate_comprehensive_scores()

        # Display results
        print(f"\nAnalysis Summary:")
        print(f"   - Total districts analyzed: {len(self.combined_data)}")
        print(f"   - Survey responses: {len(self.survey_data)}")
        print(
            f"   - Districts with current members: {len(self.combined_data[self.combined_data['member_count'] > 0])}"
        )
        print(
            f"   - SSL High-potential districts: {len(self.combined_data[self.combined_data['ssl_prediction'] == 1])}"
        )

        self.generate_recommendations()

        # Create score list for mapping
        gu_score_list = [
            {"gu": row["gu"], "score": row["final_score"]}
            for idx, row in self.final_scores.iterrows()
        ]

        return self.final_scores, gu_score_list

    # Optional: Visualize SSL decision boundaries
    def visualize_ssl_decision_space(self):
        """Visualize how SSL made its decisions"""
        if self.final_scores is None:
            print("Run analysis first!")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: SSL Score vs Member Count
        colors = [
            "red" if pred == 1 else "blue"
            for pred in self.final_scores["ssl_prediction"]
        ]
        ax1.scatter(
            self.final_scores["member_count"],
            self.final_scores["final_score"],
            c=colors,
            alpha=0.7,
            s=100,
        )
        ax1.set_xlabel("Current Member Count")
        ax1.set_ylabel("SSL Final Score")
        ax1.set_title("SSL Score vs Current Members")
        ax1.grid(True, alpha=0.3)

        # Plot 2: SSL Score vs Office Worker %
        ax2.scatter(
            self.final_scores["office_worker_pct"],
            self.final_scores["final_score"],
            c=colors,
            alpha=0.7,
            s=100,
        )
        ax2.set_xlabel("Office Worker %")
        ax2.set_ylabel("SSL Final Score")
        ax2.set_title("SSL Score vs Demographics")
        ax2.grid(True, alpha=0.3)

        # Plot 3: SSL Score vs Age 20-39 %
        ax3.scatter(
            self.final_scores["20to39_pct"],
            self.final_scores["final_score"],
            c=colors,
            alpha=0.7,
            s=100,
        )
        ax3.set_xlabel("Age 20-39 %")
        ax3.set_ylabel("SSL Final Score")
        ax3.set_title("SSL Score vs Young Adults")
        ax3.grid(True, alpha=0.3)

        # Plot 4: SSL Confidence vs Final Score
        ax4.scatter(
            self.final_scores["ssl_confidence"],
            self.final_scores["final_score"],
            c=colors,
            alpha=0.7,
            s=100,
        )
        ax4.set_xlabel("SSL Confidence")
        ax4.set_ylabel("SSL Final Score")
        ax4.set_title("SSL Confidence vs Score")
        ax4.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", label="SSL Prediction: High Potential"),
            Patch(facecolor="blue", label="SSL Prediction: Low Potential"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
        )

        plt.tight_layout()
        plt.show()

    def plot_seoul_score_map(self, gu_score_list):
        # Check the input list
        print("First 3 items in gu_score_list:", gu_score_list[:3])
        if not isinstance(gu_score_list, list) or not all(
            isinstance(d, dict) for d in gu_score_list
        ):
            raise ValueError(
                "gu_score_list must be a list of dicts with 'gu' and 'score' keys."
            )

        # Convert to DataFrame
        score_df = pd.DataFrame(gu_score_list)
        min_score = score_df["score"].min()
        max_score = score_df["score"].max()

        # Create a list of thresholds (for example 6 bins)
        threshold_scale = list(np.linspace(min_score, max_score, 6))

        # Rename columns to match geojson property keys
        score_df.rename(columns={"gu": "name"}, inplace=True)

        # Ensure score column is numeric
        score_df["score"] = pd.to_numeric(score_df["score"], errors="coerce")
        score_df = score_df.dropna(subset=["score"])  # drop rows where score is NaN

        with open("seoul_gu.geojson", encoding="utf-8") as f:
            geojson_data = json.load(f)

        # Map scores to geojson features
        score_dict = dict(zip(score_df["name"], score_df["score"]))
        for feature in geojson_data["features"]:
            gu_name = feature["properties"]["name"]
            feature["properties"]["score"] = score_dict.get(gu_name, 0)

        # Create map
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

        folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            data=score_df,
            columns=["name", "score"],
            key_on="feature.properties.name",
            fill_color="YlGnBu",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Score",
            threshold_scale=threshold_scale,
        ).add_to(m)

        m.save("seoul_score_map.html")

    import networkx as nx

    def visualize_knn_label_graph(self, X_scaled, y_pred, n_neighbors=7):
        """Visualize kNN graph with real geographical positions and Hangul labels"""

        if self.combined_data is None:
            print("Run analysis first to generate data.")
            return

        # Load GeoJSON with geometry
        gdf = gpd.read_file("seoul_gu.geojson", encoding="utf-8")
        gdf["centroid"] = gdf["geometry"].centroid
        gdf["x"] = gdf["centroid"].x
        gdf["y"] = gdf["centroid"].y

        # Create a mapping: district name -> (x, y)
        gu_to_coords = dict(zip(gdf["name"], zip(gdf["x"], gdf["y"])))

        # Get district names and coordinates
        district_names = self.combined_data["gu"].tolist()
        positions = {
            i: gu_to_coords.get(name, (0, 0)) for i, name in enumerate(district_names)
        }

        # Build kNN graph
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(X_scaled)
        similarities = np.exp(-distances)

        G = nx.Graph()
        for i, name in enumerate(district_names):
            G.add_node(i, label=name)

        for i in range(len(district_names)):
            for j_idx, j in enumerate(indices[i]):
                if i != j:
                    weight = similarities[i][j_idx]
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=weight)

        # Draw graph
        plt.figure(figsize=(12, 12))

        # IMPROVED FONT SETTING - Apply to all matplotlib elements
        try:
            # Try multiple Korean font options
            korean_fonts = [
                "Malgun Gothic",
                "맑은 고딕",
                "NanumGothic",
                "나눔고딕",
                "Arial Unicode MS",
                "Gulim",
                "굴림",
            ]

            # Find available Korean font
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            korean_font = None

            for font in korean_fonts:
                if font in available_fonts:
                    korean_font = font
                    break

            if korean_font is None:
                # Fallback: try to add font file directly
                font_paths = [
                    "C:/Windows/Fonts/malgun.ttf",
                    "C:/Windows/Fonts/gulim.ttc",
                    "/System/Library/Fonts/AppleGothic.ttf",  # macOS
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
                ]

                for font_path in font_paths:
                    try:
                        if os.path.exists(font_path):
                            fm.fontManager.addfont(font_path)
                            korean_font = fm.FontProperties(fname=font_path).get_name()
                            break
                    except:
                        continue

            if korean_font:
                # Set font for ALL matplotlib text
                plt.rcParams["font.family"] = korean_font
                plt.rcParams["font.sans-serif"] = [korean_font]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"Using Korean font: {korean_font}")
            else:
                print(
                    "Warning: No Korean font found. Korean text may not display correctly."
                )

        except Exception as e:
            print(f"Font setting error: {e}")

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos=positions, node_size=500, node_color=y_pred, cmap="coolwarm"
        )

        # Create FontProperties object for labels (additional safety)
        if korean_font:
            font_prop = fm.FontProperties(family=korean_font, size=9)
        else:
            font_prop = fm.FontProperties(size=9)

        # Add labels with explicit font properties
        labels = {i: name for i, name in enumerate(district_names)}

        # METHOD 1: Use networkx with font properties
        try:
            nx.draw_networkx_labels(
                G,
                pos=positions,
                labels=labels,
                font_size=14,
                font_family=korean_font if korean_font else "sans-serif",
            )
        except:
            # METHOD 2: Manual label drawing with explicit font control
            for node, (x, y) in positions.items():
                plt.text(
                    x,
                    y,
                    labels[node],
                    fontproperties=font_prop,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )

        # Draw edges
        edge_weights = [G[u][v]["weight"] * 10 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos=positions, width=edge_weights, alpha=0.6)

        # Set title with explicit font
        plt.title(
            "서울시 구별 kNN 그래프 (Label Propagation 기반)",
            fontproperties=font_prop if korean_font else None,
            fontsize=14,
        )

        plt.axis("off")
        plt.tight_layout()
        plt.show()


# Usage
if __name__ == "__main__":
    analyzer = SeoulDistrictAnalyzer()
    results, gu_score_list = analyzer.run_complete_analysis()

    analyzer.plot_seoul_score_map(gu_score_list)
    print("\nGenerating kNN label propagation graph...")
    # Re-run label propagation to get needed components
    X_scaled, y_pred, y_proba = analyzer.apply_semi_supervised_learning()
    analyzer.visualize_knn_label_graph(X_scaled, y_pred, n_neighbors=7)

    # Display final results table
    if results is not None:
        print("\n" + "=" * 100)
        print("FINAL RESULTS TABLE")
        print("=" * 100)
        display_cols = [
            "gu",
            "final_score",
            "demographic_score",
            "demand_score",
            "member_count",
            "office_worker_pct",
            "women_pct",
            "20to39_pct",
        ]
        print(results[display_cols].round(3).to_string(index=False))
