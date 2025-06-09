import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import sys
import io
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import folium

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
        """Apply semi-supervised learning to adjust scoring"""
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
        # Label high-demand districts (those with many members or high accessibility)
        y = np.full(len(X), -1)  # -1 for unlabeled

        # Label districts with high member count as positive
        high_member_threshold = self.combined_data["member_count"].quantile(0.75)
        high_member_mask = self.combined_data["member_count"] >= high_member_threshold
        y[high_member_mask] = 1

        # Label districts with very low member count as negative
        low_member_threshold = self.combined_data["member_count"].quantile(0.25)
        low_member_mask = self.combined_data["member_count"] <= low_member_threshold
        y[low_member_mask] = 0

        # Apply Label Propagation
        label_prop = LabelPropagation(kernel="knn", n_neighbors=7)
        label_prop.fit(
            X_scaled, y
        )  # y must contain labels, with -1 for unlabeled points
        y_pred = label_prop.predict(X_scaled)

        # Get prediction probabilities
        y_proba = label_prop.predict_proba(X_scaled)

        self.combined_data["ssl_prediction"] = y_pred
        self.combined_data["ssl_confidence"] = np.max(y_proba, axis=1)

        return X_scaled, y_pred, y_proba

    def calculate_comprehensive_scores(self):
        """Calculate final comprehensive scores using label propagation weights"""
        scaler = MinMaxScaler()

        # Define scoring components
        demographic_features = ["office_worker_pct", "women_pct", "20to39_pct"]
        demand_features = ["member_count", "accessibility_score"]

        # Create demographic score
        demo_scores = scaler.fit_transform(self.combined_data[demographic_features])
        self.combined_data["demographic_score"] = np.mean(demo_scores, axis=1)

        # Create demand score
        demand_scores = scaler.fit_transform(self.combined_data[demand_features])
        self.combined_data["demand_score"] = np.mean(demand_scores, axis=1)

        # Accessibility score (normalized)
        self.combined_data["normalized_accessibility"] = scaler.fit_transform(
            self.combined_data[["accessibility_score"]]
        ).flatten()

        # Get label propagation confidence
        ssl_confidence = self.combined_data["ssl_confidence"]

        # Calculate correlations of each score with ssl_confidence (importance)
        corr_demo = np.corrcoef(
            self.combined_data["demographic_score"], ssl_confidence
        )[0, 1]
        corr_demand = np.corrcoef(self.combined_data["demand_score"], ssl_confidence)[
            0, 1
        ]
        corr_access = np.corrcoef(
            self.combined_data["normalized_accessibility"], ssl_confidence
        )[0, 1]

        # Handle possible NaNs (if no variance)
        corrs = np.array([corr_demo, corr_demand, corr_access])
        corrs = np.nan_to_num(corrs, nan=0.0)

        # Make correlations positive (absolute), as weight magnitude
        corrs = np.abs(corrs)

        # Normalize to sum to 1
        if corrs.sum() == 0:
            weights = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            weights = corrs / corrs.sum()

        # Debug print weights
        print(f"Dynamic Weights from Label Propagation Confidence:")
        print(f"Demographic Score Weight: {weights[0]:.3f}")
        print(f"Demand Score Weight: {weights[1]:.3f}")
        print(f"Accessibility Score Weight: {weights[2]:.3f}")

        # Final comprehensive score using learned weights + ssl confidence as additional factor
        self.combined_data["final_score"] = (
            weights[0] * self.combined_data["demographic_score"]
            + weights[1] * self.combined_data["demand_score"]
            + weights[2] * self.combined_data["normalized_accessibility"]
            + 0.1 * ssl_confidence  # Keep a small factor of ssl confidence itself
        )

        # Sort by final score
        self.final_scores = self.combined_data.sort_values(
            "final_score", ascending=False
        )

        print("Comprehensive scores calculated with label propagation weights!")

    def generate_recommendations(self):
        """Generate actionable recommendations"""
        excluded_districts = ["서초구", "중구", "성동구"]

        # Filter out districts with existing clubs from recommendations
        filtered_scores = self.final_scores[
            ~self.final_scores["gu"].isin(excluded_districts)
        ]
        print("\n" + "=" * 80)
        print("SEOUL DISTRICT ANALYSIS - FINAL RECOMMENDATIONS")
        print("=" * 80)

        top_5 = filtered_scores.head(5)
        print(f"\nTOP 5 RECOMMENDED DISTRICTS:")
        print("-" * 50)

        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['gu']}")
            print(f"   Comprehensive Score: {row['final_score']:.3f}")
            print(
                f"   Demographics: Office Workers {row['office_worker_pct']:.1f}%, Women {row['women_pct']:.1f}%, Age 20-39 {row['20to39_pct']:.1f}%"
            )
            print(f"   Current Members: {row['member_count']:.0f}")
            if not pd.isna(row["travel_time_min"]):
                print(f"   Avg Travel Time: {row['travel_time_min']:.1f} minutes")
            print()

        # Analysis insights
        print("\nKEY INSIGHTS:")
        print("-" * 30)

        # Demographic insights
        high_office_worker = self.final_scores[
            self.final_scores["office_worker_pct"] > 60
        ]["gu"].tolist()
        print(f"High Office Worker Concentration: {', '.join(high_office_worker[:5])}")

        # Young adult concentration
        high_young_adult = self.final_scores[self.final_scores["20to39_pct"] > 30][
            "gu"
        ].tolist()
        print(f"High Young Adult Population: {', '.join(high_young_adult[:5])}")

        # Current service analysis
        served_districts = self.combined_data[self.combined_data["member_count"] > 0][
            "gu"
        ].tolist()
        print(f"Currently Served Districts: {', '.join(served_districts)}")

        underserved = self.final_scores[
            (self.final_scores["final_score"] > 0.5)
            & (self.final_scores["member_count"] == 0)
        ]["gu"].tolist()
        if underserved:
            print(f"High Potential Underserved: {', '.join(underserved[:3])}")

        print("\nSTRATEGIC RECOMMENDATIONS:")
        print("-" * 40)
        print("1. Primary Expansion Targets:")
        print(
            f"   - {top_5.iloc[0]['gu']}: Highest overall score ({top_5.iloc[0]['final_score']:.3f})"
        )
        print(f"   - {top_5.iloc[1]['gu']}: Strong demographics and accessibility")

        print("\n2. Location Strategy:")
        high_accessibility = self.final_scores.nlargest(3, "normalized_accessibility")[
            "gu"
        ].tolist()
        print(f"   - Best accessibility: {', '.join(high_accessibility)}")

        print("\n3. Service Optimization:")
        current_clubs = self.survey_data["closest_club_gu"].value_counts()
        for club, count in current_clubs.items():
            avg_travel = self.survey_data[self.survey_data["closest_club_gu"] == club][
                "travel_time_min"
            ].mean()
            print(f"   - {club}: {count} members, avg travel {avg_travel:.1f} min")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Seoul District Analysis...")

        if not self.load_data():
            return

        self.preprocess_survey_data()
        self.create_combined_features()
        self.apply_semi_supervised_learning()
        self.calculate_comprehensive_scores()

        # Display results
        print(f"\nAnalysis Summary:")
        print(f"   - Total districts analyzed: {len(self.combined_data)}")
        print(f"   - Survey responses: {len(self.survey_data)}")
        print(
            f"   - Districts with current members: {len(self.combined_data[self.combined_data['member_count'] > 0])}"
        )

        self.generate_recommendations()
        gu_score_list = [
            {"gu": row["gu"], "score": row["final_score"]}
            for idx, row in self.final_scores.iterrows()
        ]

        return self.final_scores, gu_score_list

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


# Usage
if __name__ == "__main__":
    analyzer = SeoulDistrictAnalyzer()
    results, gu_score_list = analyzer.run_complete_analysis()
    print(type(gu_score_list))
    analyzer.plot_seoul_score_map(gu_score_list)

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
