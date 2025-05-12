extends OptionButton

enum ThresholdTypes {THRESH_BINARY = 0, THRESH_BINARY_INV = 1, THRESH_TRUNC = 2, THRESH_TOZERO = 3, THRESH_TOZERO_INV = 4, THRESH_MASK = 7, THRESH_OTSU = 8, THRESH_TRIANGLE = 16, THRESH_DRYRUN = 128}

func _ready() -> void:
    self.item_selected.connect()
    pass

func _on_item_selected(index : int) -> void:
    #0 THRESH_BINARY = 0, 
    #1 THRESH_BINARY_INV = 1, 
    #2 THRESH_TRUNC = 2, 
    #3 THRESH_TOZERO = 3, 
    #4 THRESH_TOZERO_INV = 4, 
    #5 THRESH_MASK = 7, 
    #6 THRESH_OTSU = 8, 
    #7 THRESH_TRIANGLE = 16, 
    #8 THRESH_DRYRUN = 128
    match index:
        ThresholdTypes.:
            
        _:
            default
    pass