from sklearn.inspection import permutation_importance
import pandas as pd

def print_feature_importance(model, X_valid, y_valid):
    cols = ['cont__Age', 'cont__CabinNumber', 'cont__GroupSize', 'cont__RoomService',
 'cont__FoodCourt', 'cont__ShoppingMall', 'cont__Spa' 'cont__VRDeck',
 'cont__TotalSpending', 'cont__LuxurySpend', 'cont__EssentialSpend',
 'cont__FamilySize', 'cat__Destination', 'cat__HomePlanet', 'cat__CryoSleep',
 'cat__VIP', 'cat__CabinDeck', 'cat__CabinSide', 'cat__NoSpend',
 'cat__SoloTravel']

    X_valid_pd = pd.DataFrame(X_valid, columns=cols)
    result = permutation_importance(
        model,
        X_valid_pd,
        y_valid,
        n_repeats=10,
        random_state=0
    )
    fi = pd.DataFrame({
        'feature': X_valid_pd.columns,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)

    print(fi)
