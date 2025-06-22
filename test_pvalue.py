#!/usr/bin/env python3
"""
Test script to verify p-value calculation works correctly
"""

from earthquake_chain_analysis import EarthquakeChainAnalyzer
from earthquake_analysis import EarthquakeAnalyzer

def test_pvalue():
    """Test that p-value calculation works without errors"""
    
    print("Testing p-value calculation...")
    
    try:
        # Load and preprocess data
        analyzer = EarthquakeAnalyzer('japanearthquake_cleaned.csv')
        chain_analyzer = EarthquakeChainAnalyzer(analyzer.df)
        
        # Run aftershock analysis
        chain_analyzer.identify_aftershocks(main_shock_mag=5.0, time_window=7, distance_window=100)
        
        # Check if p-values are calculated
        if hasattr(chain_analyzer, 'omori_params') and not chain_analyzer.omori_params.empty:
            print("✅ Omori parameters calculated successfully!")
            print(f"Number of mainshocks analyzed: {len(chain_analyzer.omori_params)}")
            
            # Check p-value column exists
            if 'p_value' in chain_analyzer.omori_params.columns:
                print("✅ P-value column exists!")
                
                # Show sample p-values
                print("\nSample p-values:")
                sample_pvalues = chain_analyzer.omori_params[['main_shock_time', 'main_shock_mag', 'p_value']].head()
                print(sample_pvalues)
                
                # Check p-value statistics
                pvalues = chain_analyzer.omori_params['p_value']
                print(f"\nP-value statistics:")
                print(f"Min: {pvalues.min():.4f}")
                print(f"Max: {pvalues.max():.4f}")
                print(f"Mean: {pvalues.mean():.4f}")
                
                # Count good fits (p < 0.05)
                good_fits = (pvalues < 0.05).sum()
                total_fits = len(pvalues)
                print(f"Good fits (p < 0.05): {good_fits}/{total_fits} ({good_fits/total_fits*100:.1f}%)")
                
                print("\n✅ P-value calculation working correctly!")
                
            else:
                print("❌ P-value column missing!")
                
        else:
            print("❌ No Omori parameters found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pvalue() 