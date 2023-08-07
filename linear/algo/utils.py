def tag_unique(df):
    seen = []
    tags = []

    for row in df.iterrows():
        if str(row[1].values) not in seen:
            seen.append(str(row[1].values))
            tags.append(len(seen))
        else:
            tags.append(seen.index(str(row[1].values)) + 1)

    df['unique_tags'] = tags
    return df