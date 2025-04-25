import streamlit as st
import pandas as pd
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_folium import st_folium
from collections import deque
from typing import Dict, List, Optional, Tuple


class HealthDataProcessor:
    """
    Handles loading and processing of health and geographic data.
    
    Attributes:
        health_df (pd.DataFrame): DataFrame containing health metrics (PLACES) data
        geo_df (pd.DataFrame): DataFrame containing geographic county (latitude/ longitude) data
        merged_df (pd.DataFrame): Merged health and geographic data
        health_metrics (pd.DataFrame): Pivoted health metrics with normalized values
        similarity_matrix (np.ndarray): Cosine similarity matrix of counties
        graph (Dict[str, List[str]]): Graph representation of county connections
    """
    
    def __init__(self, health_data_path: str, geo_data_path: str):
        """
        Initialize the data processor with file paths.
        
        Args:
            health_data_path: Path to the health metrics CSV file
            geo_data_path: Path to the geographic county data CSV file
        """
        self.health_df = pd.read_csv(health_data_path)
        self.geo_df = pd.read_csv(geo_data_path)
        self.merged_df = None
        self.health_metrics = None
        self.similarity_matrix = None
        self.graph = None
        self._process_data()
    
    def _process_data(self) -> None:
        """Process and merge the health and geographic data."""
        # Preprocess FIPS codes
        self.health_df['county_fips'] = self.health_df['LocationID'].astype(str).str.zfill(5)
        self.geo_df['county_fips'] = self.geo_df['county_fips'].astype(str).str.zfill(5)
        
        # Pivot health data
        health_pivot = self.health_df.pivot_table(
            index='county_fips',
            columns='Measure',
            values='Data_Value',
            aggfunc='mean'
        ).dropna()
        
        # Merge with geo data
        self.merged_df = health_pivot.merge(
            self.geo_df[['county_fips', 'lat', 'lng', 'county_full', 'state_name']],
            on='county_fips',
            how='left'
        )
        
        # Normalize health metrics
        self.health_metrics = self.merged_df.drop(
            columns=['lat', 'lng', 'county_full', 'state_name']
        ).set_index('county_fips')
        
        scaler = StandardScaler()
        normalized = scaler.fit_transform(self.health_metrics)
        county_fips_list = self.merged_df['county_fips'].tolist()
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(normalized)
        
        # Build graph as adjacency list
        self._build_graph(county_fips_list)
    
    def _build_graph(self, county_fips_list: List[str], threshold: float = 0.9) -> None:
        """
        Build a graph of county connections based on similarity threshold.
        
        Args:
            county_fips_list: List of county FIPS codes
            threshold: Similarity threshold for creating edges (default: 0.9)
        """
        self.graph = {fips: [] for fips in county_fips_list}
        
        for i in range(len(county_fips_list)):
            for j in range(i + 1, len(county_fips_list)):
                sim = self.similarity_matrix[i][j]
                if sim >= threshold:
                    self.graph[county_fips_list[i]].append(county_fips_list[j])
                    self.graph[county_fips_list[j]].append(county_fips_list[i])
    
    def get_fips_from_name(self, state: str, county: str) -> Optional[str]:
        """
        Get FIPS code from state and county names.
        
        Args:
            state: State name
            county: County name
            
        Returns:
            FIPS code if found, None otherwise
        """
        name_to_fips = {
            (row['state_name'].lower(), row['county_full'].lower()): row['county_fips']
            for _, row in self.merged_df.iterrows()
            if pd.notna(row['state_name']) and pd.notna(row['county_full'])
        }
        return name_to_fips.get((state.lower(), county.lower()), None)
    
    def get_county_profile(self, fips: str) -> Optional[pd.Series]:
        """
        Get county profile by FIPS code.
        
        Args:
            fips: County FIPS code
            
        Returns:
            County profile as a Series if found, None otherwise
        """
        matches = self.merged_df[self.merged_df['county_fips'] == fips]
        return matches.iloc[0] if not matches.empty else None


class CountyNetworkAnalyzer:
    """
    Provides analysis methods for the county health network.
    
    Attributes:
        data_processor (HealthDataProcessor): Instance of HealthDataProcessor
    """
    
    def __init__(self, data_processor: HealthDataProcessor):
        """
        Initialize with a HealthDataProcessor instance.
        
        Args:
            data_processor: HealthDataProcessor instance
        """
        self.data_processor = data_processor
    
    def bfs_shortest_path(self, start: str, goal: str) -> Optional[List[str]]:
        """
        Find the shortest path between two counties using BFS.
        
        Args:
            start: Starting county FIPS code
            goal: Target county FIPS code
            
        Returns:
            List of FIPS codes representing the path if found, None otherwise
        """
        graph = self.data_processor.graph
        visited = set()
        queue = deque([[start]])
        
        while queue:
            path = queue.popleft()
            node = path[-1]
            
            if node == goal:
                return path
            elif node not in visited:
                for neighbor in graph.get(node, []):
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                visited.add(node)
        
        return None
    
    def calculate_degree_centrality(self) -> Dict[str, float]:
        """
        Calculate degree centrality for all counties in the network.
        
        Returns:
            Dictionary mapping FIPS codes to their degree centrality values
        """
        graph = self.data_processor.graph
        return {
            node: len(neigh) / (len(graph) - 1)
            for node, neigh in graph.items()
        }
    
    def get_top_hubs(self, top_n: int = 5) -> List[Dict[str, str]]:
        """
        Get the top network hubs by degree centrality.
        
        Args:
            top_n: Number of top hubs to return (default: 5)
            
        Returns:
            List of dictionaries containing hub information
        """
        centrality = self.calculate_degree_centrality()
        centrality_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        hub_info = []
        for fips, score in centrality_sorted:
            row = self.data_processor.merged_df[self.data_processor.merged_df['county_fips'] == fips].iloc[0]
            hub_info.append({
                "County": row['county_full'],
                "State": row['state_name'],
                "Centrality": round(score, 4)
            })
        
        return hub_info
    
    def get_similar_counties_by_metric(self, fips: str, metric: str, top_n: int = 5) -> List[Dict[str, str]]:
        """
        Get counties most similar to the given county for a specific metric.
        
        Args:
            fips: Reference county FIPS code
            metric: Health metric to compare
            top_n: Number of similar counties to return (default: 5)
            
        Returns:
            List of dictionaries containing similar county information
        """
        health_metrics = self.data_processor.health_metrics
        metric_vals = health_metrics[metric]
        selected_val = metric_vals[fips]
        metric_diff = abs(metric_vals - selected_val)
        similar_counties = metric_diff.sort_values().drop(index=fips).head(top_n)
        
        similar_info = []
        for similar_fips, diff in similar_counties.items():
            county_row = self.data_processor.merged_df[
                self.data_processor.merged_df['county_fips'] == similar_fips
            ].iloc[0]
            val = health_metrics.loc[similar_fips][metric]
            similar_info.append({
                "County": county_row['county_full'],
                "State": county_row['state_name'],
                "Value": f"{val:.2f}%",
                "Difference": f"{diff:.2f}"
            })
        
        return similar_info


class CountyVisualizer:
    """
    Handles visualization of county data and network relationships.
    
    Attributes:
        data_processor (HealthDataProcessor): Instance of HealthDataProcessor
        network_analyzer (CountyNetworkAnalyzer): Instance of CountyNetworkAnalyzer
    """
    
    def __init__(self, data_processor: HealthDataProcessor, network_analyzer: CountyNetworkAnalyzer):
        """
        Initialize with data processor and network analyzer.
        
        Args:
            data_processor: HealthDataProcessor instance
            network_analyzer: CountyNetworkAnalyzer instance
        """
        self.data_processor = data_processor
        self.network_analyzer = network_analyzer
    
    def create_county_map(self, fips: str, neighbors: List[str]) -> folium.Map:
        """
        Create a Folium map showing the county and its connected neighbors.
        
        Args:
            fips: Center county FIPS code
            neighbors: List of neighboring county FIPS codes
            
        Returns:
            Configured Folium map
        """
        profile = self.data_processor.get_county_profile(fips)
        m = folium.Map(location=[profile['lat'], profile['lng']], zoom_start=7)
        
        # Add main county marker
        folium.Marker(
            location=[profile['lat'], profile['lng']],
            tooltip=f"{profile['county_full']}, {profile['state_name']}",
            popup=f"FIPS: {profile['county_fips']}"
        ).add_to(m)
        
        # Add neighbor markers
        for neighbor in neighbors:
            neighbor_row = self.data_processor.get_county_profile(neighbor)
            folium.Marker(
                location=[neighbor_row['lat'], neighbor_row['lng']],
                icon=folium.Icon(color='green'),
                tooltip=f"{neighbor_row['county_full']}, {neighbor_row['state_name']}"
            ).add_to(m)
        
        return m
    
    def create_path_map(self, path: List[str]) -> folium.Map:
        """
        Create a Folium map showing a path between counties.
        
        Args:
            path: List of FIPS codes representing the path
            
        Returns:
            Configured Folium map with path visualization
        """
        if not path:
            return None
        
        # Get first county for initial map location
        first_county = self.data_processor.get_county_profile(path[0])
        path_map = folium.Map(
            location=[first_county['lat'], first_county['lng']],
            zoom_start=6
        )
        
        path_coords = []
        for fips in path:
            row = self.data_processor.get_county_profile(fips)
            path_coords.append((row['lat'], row['lng']))
            folium.Marker(
                [row['lat'], row['lng']],
                tooltip=f"{row['county_full']}, {row['state_name']}"
            ).add_to(path_map)
        
        folium.PolyLine(path_coords, color="blue", weight=4).add_to(path_map)
        return path_map


class CountyHealthApp:
    """
    Main Streamlit application class for the County Health Network Explorer.
    
    Attributes:
        data_processor (HealthDataProcessor): Instance of HealthDataProcessor
        network_analyzer (CountyNetworkAnalyzer): Instance of CountyNetworkAnalyzer
        visualizer (CountyVisualizer): Instance of CountyVisualizer
    """
    
    def __init__(self, health_data_path: str, geo_data_path: str):
        """
        Initialize the application with data file paths.
        
        Args:
            health_data_path: Path to health metrics CSV
            geo_data_path: Path to geographic county data CSV
        """
        self.data_processor = HealthDataProcessor(health_data_path, geo_data_path)
        self.network_analyzer = CountyNetworkAnalyzer(self.data_processor)
        self.visualizer = CountyVisualizer(self.data_processor, self.network_analyzer)
    
    def run(self) -> None:
        """Run the Streamlit application."""
        st.set_page_config(layout="wide")
        st.title("County Health Network Explorer")
        
        # Get default selections
        states = sorted(self.data_processor.merged_df['state_name'].dropna().unique())
        default_state = self._get_default_selection(states, "Michigan")
        
        # County selection
        selected_state, selected_county = self._render_county_selection(states, default_state)
        selected_fips = self.data_processor.get_fips_from_name(selected_state, selected_county)
        
        if not selected_fips:
            st.warning("County not found.")
            return
        
        # Show county profile
        self._render_county_profile(selected_fips)
    
    def _get_default_selection(self, options: List[str], preferred: str) -> str:
        """
        Get default selection from a list of options.
        
        Args:
            options: List of available options
            preferred: Preferred default option
            
        Returns:
            The default option to use
        """
        return preferred if preferred in options else options[0]
    
    def _render_county_selection(self, states: List[str], default_state: str) -> Tuple[str, str]:
        """
        Render the county selection widgets.
        
        Args:
            states: List of available states
            default_state: Default state to select
            
        Returns:
            Tuple of (selected_state, selected_county)
        """
        col1, col2 = st.columns(2)
        
        with col1:
            # Set default state index
            default_state_index = states.index(default_state) if default_state in states else 0
            selected_state = st.selectbox("Select a state", states, index=default_state_index)
        
        with col2:
            # Set default county index within selected state
            filtered_df = self.data_processor.merged_df[
                self.data_processor.merged_df['state_name'] == selected_state
            ]
            counties = sorted(filtered_df['county_full'].dropna().unique())
            default_county = self._get_default_selection(counties, "Washtenaw County")
            default_county_index = counties.index(default_county) if default_county in counties else 0
            selected_county = st.selectbox("Select a county", counties, index=default_county_index)
        
        return selected_state, selected_county
    
    def _render_county_profile(self, fips: str) -> None:
        """
        Render the county profile and all related visualizations.
        
        Args:
            fips: County FIPS code to display
        """
        profile = self.data_processor.get_county_profile(fips)
        neighbors = self.data_processor.graph.get(fips, [])
        degree_centrality = self.network_analyzer.calculate_degree_centrality()
        centrality_value = degree_centrality[fips]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic county info
            st.markdown(f"## {profile['county_full']}, {profile['state_name']}")
            st.markdown(f"### Connected Counties: {len(neighbors)}")
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric(label="Latitude", value=f"{profile['lat']:.4f}")
            with col4:
                st.metric(label="Longitude", value=f"{profile['lng']:.4f}")
            with col5:
                st.metric(label="Degree Centrality", value=f"{centrality_value:.4f}")
            
            # County map
            st.markdown("### County Map")
            county_map = self.visualizer.create_county_map(fips, neighbors)
            st_folium(county_map, width=700, height=500)
        
        with col2:
            # Network hubs
            st.markdown("### Top Network Hubs (by Degree Centrality)")
            hub_info = self.network_analyzer.get_top_hubs()
            st.table(pd.DataFrame(hub_info))
            
            # Health metrics
            st.markdown("### Key Health Metrics")
            metrics = self.data_processor.health_metrics.loc[fips].to_frame().reset_index()
            metrics.columns = ["Health Metric", "Population Percent Affected (County Level)"]
            metrics['Population Percent Affected (County Level)'] = metrics[
                'Population Percent Affected (County Level)'
            ].apply(lambda x: f"{x:.2f}%")
            st.dataframe(metrics, height=400)
        
        # Similar health metrics with connected county
        self._render_connected_county_comparison(fips, neighbors)
        
        # Path between counties
        self._render_county_path_finder(fips, profile)
    
    def _render_connected_county_comparison(self, fips: str, neighbors: List[str]) -> None:
        """
        Render comparison with connected counties.
        
        Args:
            fips: Reference county FIPS code
            neighbors: List of neighboring county FIPS codes
        """
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Similar Metrics with a Connected County")
            
            if neighbors:
                selected_neighbor_fips = st.selectbox(
                    "Select a connected county to compare health metrics",
                    options=neighbors,
                    format_func=lambda f: self._format_county_name(f)
                )
                
                self._show_metric_comparison(fips, selected_neighbor_fips)
            else:
                st.write("No connected counties to compare.")
        
        with col2:
            st.markdown("### Find Counties with Similar Metric")
            all_metrics = self.data_processor.health_metrics.columns.tolist()
            selected_metric = st.selectbox("Choose a health metric:", all_metrics)
            
            similar_info = self.network_analyzer.get_similar_counties_by_metric(fips, selected_metric)
            profile = self.data_processor.get_county_profile(fips)
            
            st.markdown(
                f"**Counties most similar in {selected_metric} to "
                f"{profile['county_full']}, {profile['state_name']}**"
            )
            
            for info in similar_info:
                st.write(
                    f"- {info['County']}, {info['State']}: {info['Value']} "
                    f"(difference: {info['Difference']})"
                )
    
    def _format_county_name(self, fips: str) -> str:
        """
        Format county name for display.
        
        Args:
            fips: County FIPS code
            
        Returns:
            Formatted county name string
        """
        row = self.data_processor.get_county_profile(fips)
        return f"{row['county_full']}, {row['state_name']}"
    
    def _show_metric_comparison(self, fips1: str, fips2: str) -> None:
        """
        Show comparison of health metrics between two counties.
        
        Args:
            fips1: First county FIPS code
            fips2: Second county FIPS code
        """
        health_metrics = self.data_processor.health_metrics
        diffs = abs(health_metrics.loc[fips1] - health_metrics.loc[fips2])
        sorted_diffs = diffs.sort_values()
        
        county2_name = self._format_county_name(fips2)
        st.markdown(f"**Comparing with {county2_name}**")
        
        for metric in sorted_diffs.index[:5]:
            v1 = health_metrics.loc[fips1][metric]
            v2 = health_metrics.loc[fips2][metric]
            diff = sorted_diffs[metric]
            st.write(f"- {metric}: {v1:.2f}% vs {v2:.2f}% (difference: {diff:.2f})")
    
    def _render_county_path_finder(self, start_fips: str, start_profile: pd.Series) -> None:
        """
        Render the path finder between counties.
        
        Args:
            start_fips: Starting county FIPS code
            start_profile: Starting county profile data
        """
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Path Between Two Counties")
            states = sorted(self.data_processor.merged_df['state_name'].dropna().unique())
            
            # Show start county (disabled)
            state_idx = states.index(start_profile['state_name']) 
            st.selectbox(
                "Start State (selected)", 
                states, 
                index=state_idx, 
                disabled=True
            )
            
            counties = sorted(self.data_processor.merged_df[
                self.data_processor.merged_df['state_name'] == start_profile['state_name']
            ]['county_full'].dropna().unique())
            county_idx = counties.index(start_profile['county_full']) 
            st.selectbox(
                "Start County (selected)", 
                counties, 
                index=county_idx, 
                disabled=True
            )
            
            # Select end county with Michigan and Washtenaw County as defaults
            default_state = "Michigan"
            state_b = st.selectbox(
                "End State", 
                states, 
                key="state_b",
                index=states.index(default_state) if default_state in states else 0
            )
            
            filtered_df_b = self.data_processor.merged_df[
                self.data_processor.merged_df['state_name'] == state_b
            ]
            counties_b = sorted(filtered_df_b['county_full'].dropna().unique())
            
            default_county = "Washtenaw County"
            county_b_index = counties_b.index(default_county) if default_county in counties_b else 0
            county_b = st.selectbox(
                "End County", 
                counties_b, 
                key="county_b",
                index=county_b_index
            )
            
            fips_b = self.data_processor.get_fips_from_name(state_b, county_b)
            
            if fips_b:
                path = self.network_analyzer.bfs_shortest_path(start_fips, fips_b)
                
                if path:
                    st.success("Path Found:")
                    path_names = [self._format_county_name(f) for f in path]
                    st.write(" → ".join(path_names))
                    
                    with col2:
                        # Map with path
                        st.markdown("### Path Map")
                        path_map = self.visualizer.create_path_map(path)
                        st_folium(path_map, width=700, height=500)
                else:
                    st.error("No path found between selected counties.")

        if fips_b and path:
            self._render_path_metric_comparisons(path)
    
    def _render_path_metric_comparisons(self, path: List[str]) -> None:
        """
        Render metric comparisons along a path.
        
        Args:
            path: List of FIPS codes representing the path
        """
        st.markdown("### Shared Health Metric Similarities Along the Path")
        cols = st.columns(2)
        health_metrics = self.data_processor.health_metrics

        if len(path) < 2:
            st.write("Nothing to Show")
        
        for i in range(len(path) - 1):
            f1, f2 = path[i], path[i + 1]
            diffs = abs(health_metrics.loc[f1] - health_metrics.loc[f2])
            top_similar = diffs.nsmallest(3)
            
            c1 = self.data_processor.get_county_profile(f1)
            c2 = self.data_processor.get_county_profile(f2)
            
            with cols[i % 2]:
                st.markdown(f"**{c1['county_full']}, {c1['state_name']} → {c2['county_full']}, {c2['state_name']}**")
                for metric, diff in top_similar.items():
                    v1 = health_metrics.loc[f1][metric]
                    v2 = health_metrics.loc[f2][metric]
                    st.markdown(f"- {metric}: {v1:.2f}% vs {v2:.2f}%  \n(difference: {diff:.2f})")


# Main execution
if __name__ == "__main__":
    app = CountyHealthApp("data/PLACES.csv", "data/uscounties.csv")
    app.run()