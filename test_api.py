def split_list_into_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # Calculate chunk size and remainder
    k, m = divmod(len(lst), n)
    # Create chunks: first m chunks have size k+1, remaining have size k
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

lst = [str(x) + "/pds" for x in range(20)]

l = split_list_into_chunks(lst, 3)
print(len(l))
print(l[0])
print(l[1])
print(l[2])
