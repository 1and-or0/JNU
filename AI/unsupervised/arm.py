import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# CSV 파일 경로
file_path = 'store_data.csv'

try:
    # 데이터 로딩 (헤더 없음)
    df = pd.read_csv(file_path, header=None)

    # 트랜잭션 데이터로 변환
    transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
    print(type(transactions))

    # One-hot Encoding
    encoder = TransactionEncoder()
    te_array = encoder.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=encoder.columns_)
    print(df_encoded.info())
    
    # 빈발 항목 집합 추출
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    print(" Frequent Itemsets:")
    print(frequent_itemsets.sort_values(by="support", ascending=False))

    # 연관 규칙 생성
    min_confidence = 0.1
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Lift를 기준으로 내림차순 정렬
    rules_sorted = rules.sort_values(by="lift", ascending=False)
    
    # 결과 출력
    print("\n Association Rules:")
    print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

except FileNotFoundError:
    print(f"[ERROR] '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"[ERROR] 예외 발생: {e}")
