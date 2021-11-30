### Dependencies

```
pip3 install -r requirements.txt
```
If it does not work substitute *pip3* with *pip*

### Run

```
./solve.py /tmp/santa_solution.csv
./check.py /tmp/santa_solution.csv /tmp/map.html
```

### Tests
test with 1'000 gifts:
nearest neighbor:   308'497'451
beam search:	    261'067'295
vertical lines:	    316'515'966

test with 10'000 gifts:
nearest neighbor:   1'741'696'874
vertical lines:	    1'373'432'439

test with 100'000 gifts:
nearest neighbor:   13'995'792'965
vertical lines:	    12'664'104'995
