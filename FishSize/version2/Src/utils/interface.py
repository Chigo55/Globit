from abc import abstractmethod, ABC


class IModelLoader(ABC):
    """모델 로더 인터페이스."""

    @abstractmethod
    def load_model(self):
        """모델을 로드하는 추상 메소드입니다. 모델의 가중치와 구조 등 필요한 정보를 불러와 반환해야 합니다."""
        pass


class IDataLoader(ABC):
    """데이터 로더 인터페이스."""

    @abstractmethod
    def load_data(self):
        """데이터를 로드하는 추상 메소드입니다. 학습이나 예측에 필요한 데이터를 불러와 반환해야 합니다."""
        pass


class IDataPredictor(ABC):
    """데이터 프리딕터 인터페이스."""

    @abstractmethod
    def predict_data(self, data):
        """입력 데이터를 기반으로 예측을 수행하는 추상 메소드입니다.

        Args:
            data: 예측에 사용될 입력 데이터.

        Returns:
            예측 결과를 반환합니다.
        """
        pass


class ISizeEstimator(ABC):
    """크기 추정기 인터페이스."""

    @abstractmethod
    def estimate_size(self, data):
        """입력 데이터와 예측 결과를 활용하여 크기를 추정하는 추상 메소드입니다.

        Args:
            data: 크기 추정에 사용될 원본 데이터.

        Returns:
            추정된 크기를 반환합니다.
        """
        pass


class IVisualizer(ABC):
    """시각화 인터페이스."""

    @abstractmethod
    def draw(self, data):
        """데이터와 예측 결과를 바탕으로 시각적 표현을 그리는 추상 메소드입니다.

        Args:
            data: 시각화에 사용할 데이터.

        Returns:
            시각적 표현이 추가된 데이터를 반환합니다.
        """
        pass

    @abstractmethod
    def visualize(self, data):
        """입력 데이터를 다양한 방식으로 시각화하는 추상 메소드입니다.

        Args:
            data: 시각화할 데이터.
        """
        pass


class IResultSaver(ABC):
    """결과 저장기 인터페이스."""

    @abstractmethod
    def save(self, data):
        """결과 데이터를 지정된 포맷이나 위치에 저장하는 추상 메소드입니다.

        Args:
            data: 저장할 결과 데이터.
        """
        pass

    @abstractmethod
    def save_txt(self, data):
        """결과 데이터를 텍스트 파일 형태로 저장하는 추상 메소드입니다.

        Args:
            data: 텍스트 파일로 저장할 결과 데이터.
        """
        pass
