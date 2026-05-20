MEGNET_NUMERICAL_TO_STRING: dict[int, str] = {
    0: "brain/other",
    1: "eye blink",
    2: "heart beat",
    3: "eye movement",
}

MEGNET_STRING_TO_NUMERICAL: dict[str, int] = {
    val: key for key, val in MEGNET_NUMERICAL_TO_STRING.items()
}
