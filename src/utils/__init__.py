def check_no_overlap(arr1, arr2, arr3):
    """
    3つの配列の中身に被り（重複）がないことを確認する関数。
    """
    # 3つの配列をセットに変換して重複を検出
    set1 = set(arr1)
    set2 = set(arr2)
    set3 = set(arr3)
    
    # 各セット同士に重複がないか確認
    if set1.isdisjoint(set2) and set1.isdisjoint(set3) and set2.isdisjoint(set3):
        return True
    else:
        return False