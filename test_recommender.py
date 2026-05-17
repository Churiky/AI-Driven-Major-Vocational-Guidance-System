from src.university_recommender import UniversityRecommender

recommender = UniversityRecommender("data/diem_chuan_all.csv")
print("Total rows loaded:", len(recommender.df))

# Check some sample recommendations
scores = {
    "toan": 8, "van": 8, "anh": 8, "ly": 8, "hoa": 8, "sinh": 8, 
    "su": 8, "dia": 8, "gdcd": 8, "khtn": 8, "khxh": 8, "ngoai_ngu": 8
}

res = recommender.recommend("CongNgheThongTin", scores, preferred_method=1)
if res:
    print("Found", len(res), "results")
    print(res[0]['Truong'], res[0]['Nganh'], res[0]['Cutoff'], res[0]['History'])
else:
    print("No results found")
