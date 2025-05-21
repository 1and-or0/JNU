from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# data = pd.read_csv("./store_data.csv")

# CSV 파일 경로 설정
file_path = './store_data.csv' 

try:
    # CSV 파일 읽기 (헤더 없음)
    df = pd.read_csv(file_path, header=None)

    # 트랜잭션 데이터로 변환 (one-hot encoding)
    def to_transaction(row):
        return [item.strip() for item in row.dropna()]

    transactions = df.apply(to_transaction, axis=1).tolist()
    print(transactions)
    
    # Apriori 알고리즘을 사용하여 frequent itemsets 찾기
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # 지지도(support) 임계값 설정 (예시: 0.2)
    support_threshold = 0.2
    frequent_itemsets = apriori(df_encoded, min_support=support_threshold, use_colnames=True)

    # 연관 규칙 생성
    # 신뢰도(confidence) 임계값 설정 
    confidence_threshold = 0.7
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_threshold)

    # 결과 출력
    print("Frequent Itemsets:")
    print(frequent_itemsets)
    print("\nAssociation Rules:")
    print(rules)

except FileNotFoundError:
    print(f"Error: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
except Exception as e:
    print(f"Error 발생: {e}")