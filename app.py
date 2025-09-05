import os
import tempfile

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="실시간 소음 제거 시스템",
    page_icon="🔊",
    layout="wide",
)

# 제목 및 소개
st.title("파동의 상쇄 간섭을 활용한 지능형 소음 제거 시스템")
st.markdown("""
이 애플리케이션은 파동의 상쇄 간섭 원리와 푸리에 변환을 활용하여 오디오에서 소음을 제거합니다.
""")

# 사이드바 설정
with st.sidebar:
    st.title("소음 제거 설정")

    # 소음 제거 강도 설정
    noise_reduction_strength = st.slider(
        "소음 제거 강도",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="값이 클수록 더 많은 소음이 제거되지만, 원본 신호도 더 많이 영향을 받을 수 있습니다.",
    )

    # 주파수 범위 설정
    freq_range = st.slider(
        "관심 주파수 범위 (Hz)",
        min_value=0,
        max_value=8000,
        value=(100, 4000),
        step=100,
        help="분석 및 필터링할 주파수 범위를 설정합니다.",
    )


# 오디오 처리 클래스
class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # STFT 파라미터 캐싱
        self.n_fft = 2048
        self.hop_length = self.n_fft // 4

    def load_audio(self, audio_bytes):
        """오디오 바이트 데이터를 numpy 배열로 변환"""
        try:
            # 스트림릿의 UploadedFile 객체인 경우 바이트로 변환
            if hasattr(audio_bytes, "read"):
                audio_bytes = audio_bytes.read()

            # 임시 파일에 오디오 데이터 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(audio_bytes)
                tmpfile_path = tmpfile.name

            # 오디오 파일 읽기
            audio, sr = librosa.load(tmpfile_path, sr=self.sample_rate, mono=True)

            # 임시 파일 삭제
            os.remove(tmpfile_path)

            return audio, sr
        except Exception as e:
            st.error(f"오디오 로딩 중 오류 발생: {e}")
            return None, None

    def plot_waveform(self, audio, sr):
        """오디오 파형 시각화"""
        # 시간 배열 생성 (50ms 간격으로 샘플링하여 데이터 포인트 수 감소)
        sample_rate = max(1, int(len(audio) / 500))  # 최대 500 포인트
        t = np.arange(0, len(audio), sample_rate) / sr
        y = audio[::sample_rate]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                name="오디오 파형",
                line=dict(color="blue", width=1),
            )
        )
        fig.update_layout(
            title="오디오 파형 (시간-진폭)",
            xaxis_title="시간 (초)",
            yaxis_title="진폭",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    def plot_spectrogram(self, audio, sr):
        """스펙트로그램 시각화"""
        # STFT 계산
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # 주파수 범위 제한 (freq_range 범위만 표시)
        max_freq_idx = min(int(freq_range[1] * self.n_fft / sr) + 1, S_db.shape[0])
        S_db_cropped = S_db[:max_freq_idx]

        # 시간축 데이터 포인트 감소
        time_downsample = max(
            1, int(S_db_cropped.shape[1] / 200)
        )  # 최대 200 시간 포인트
        S_db_cropped = S_db_cropped[:, ::time_downsample]

        # 플로틀리로 스펙트로그램 그리기
        fig = go.Figure(
            data=go.Heatmap(
                z=S_db_cropped,
                colorscale="Viridis",
                colorbar=dict(title="dB"),
            )
        )
        fig.update_layout(
            title="스펙트로그램 (시간-주파수)",
            xaxis_title="시간 (s)",
            yaxis_title="주파수 (Hz)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    def plot_fft(self, audio, sr):
        """FFT 결과 시각화 (최적화)"""
        # FFT 계산
        n = len(audio)
        fft_result = np.fft.rfft(audio)
        magnitude = np.abs(fft_result)
        frequency = np.fft.rfftfreq(n, 1 / sr)

        # 관심 주파수 범위만 표시하고 데이터 포인트 감소
        mask = frequency <= freq_range[1]
        freq_reduced = frequency[mask]
        mag_reduced = magnitude[mask]

        # 데이터 포인트 더 줄이기
        sample_rate = max(1, int(len(freq_reduced) / 300))  # 최대 300 포인트
        freq_reduced = freq_reduced[::sample_rate]
        mag_reduced = mag_reduced[::sample_rate]

        # 플로틀리로 FFT 그래프 그리기
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freq_reduced,
                y=mag_reduced,
                mode="lines",
                name="주파수 스펙트럼",
                line=dict(color="green", width=1.5),
            )
        )
        fig.update_layout(
            title="주파수 스펙트럼 (FFT)",
            xaxis_title="주파수 (Hz)",
            yaxis_title="진폭",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    def process_audio(self, audio, sr, strength):
        """스펙트럼 차감법을 사용하여 소음 제거 처리"""
        # STFT 계산
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = librosa.magphase(stft)

        # 노이즈 프로파일 추정 (처음 0.5초 동안의 데이터를 노이즈로 가정)
        noise_idx = int(0.5 * sr / self.hop_length) + 1
        noise_profile = np.mean(magnitude[:, :noise_idx], axis=1, keepdims=True)

        # 스펙트럼 차감
        magnitude_reduced = np.maximum(
            magnitude - strength * noise_profile,
            0.01 * magnitude,  # 음수 값 방지
        )

        # 역변환
        audio_reduced = librosa.istft(
            magnitude_reduced * phase, hop_length=self.hop_length
        )
        return audio_reduced

    def save_audio(self, audio, sr):
        """오디오를 바이트 형식으로 저장"""
        try:
            # 오디오 정규화
            audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9

            # 임시 파일에 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(tmpfile.name, audio, sr)
                # 파일 읽기
                with open(tmpfile.name, "rb") as f:
                    wav_bytes = f.read()
            # 임시 파일 삭제
            os.remove(tmpfile.name)
            return wav_bytes
        except Exception as e:
            st.error(f"오디오 저장 중 오류 발생: {e}")
            return None


# 오디오 프로세서 인스턴스 생성
processor = AudioProcessor()

# 메인 앱 UI
st.subheader("오디오 입력")

# 오디오 입력 위젯
audio_input = st.audio_input("오디오 녹음", key="audio_recorder")

# 처리 버튼
process_clicked = st.button("소음 제거 처리", key="process_button")

# 오디오 데이터 처리
if process_clicked:
    with st.spinner("오디오 처리 중..."):
        # 오디오 데이터 가져오기
        if audio_input:
            # 오디오 데이터 로드
            audio, sr = processor.load_audio(audio_input)

            if audio is not None:
                # 결과를 저장할 컨테이너 생성
                results = st.container()

                # 소음 제거 적용
                with st.status("소음 제거 처리 중...", expanded=True) as status:
                    st.write("원본 오디오 분석 중...")
                    audio_cleaned = processor.process_audio(
                        audio, sr, noise_reduction_strength
                    )
                    st.write("처리된 오디오 저장 중...")
                    cleaned_audio_bytes = processor.save_audio(audio_cleaned, sr)
                    status.update(label="처리 완료!", state="complete")

                # 탭 생성
                tab1, tab2, tab3 = st.tabs(["파형 분석", "주파수 분석", "스펙트로그램"])

                with tab1:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("### 원본 오디오")
                        st.audio(audio_input, format="audio/wav")
                        with st.spinner("파형 그래프 생성 중..."):
                            st.plotly_chart(
                                processor.plot_waveform(audio, sr),
                                use_container_width=True,
                            )
                    with col_b:
                        st.markdown("### 소음 제거 후")
                        if cleaned_audio_bytes:
                            st.audio(cleaned_audio_bytes, format="audio/wav")
                            with st.spinner("파형 그래프 생성 중..."):
                                st.plotly_chart(
                                    processor.plot_waveform(audio_cleaned, sr),
                                    use_container_width=True,
                                )

                with tab2:
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.markdown("### 원본 오디오")
                        with st.spinner("주파수 그래프 생성 중..."):
                            st.plotly_chart(
                                processor.plot_fft(audio, sr), use_container_width=True
                            )
                    with col_d:
                        st.markdown("### 소음 제거 후")
                        with st.spinner("주파수 그래프 생성 중..."):
                            st.plotly_chart(
                                processor.plot_fft(audio_cleaned, sr),
                                use_container_width=True,
                            )

                with tab3:
                    col_e, col_f = st.columns(2)
                    with col_e:
                        st.markdown("### 원본 오디오")
                        with st.spinner("스펙트로그램 생성 중..."):
                            st.plotly_chart(
                                processor.plot_spectrogram(audio, sr),
                                use_container_width=True,
                            )
                    with col_f:
                        st.markdown("### 소음 제거 후")
                        with st.spinner("스펙트로그램 생성 중..."):
                            st.plotly_chart(
                                processor.plot_spectrogram(audio_cleaned, sr),
                                use_container_width=True,
                            )

                # 다운로드 버튼
                if cleaned_audio_bytes:
                    st.download_button(
                        label="처리된 오디오 다운로드",
                        data=cleaned_audio_bytes,
                        file_name="noise_reduced_audio.wav",
                        mime="audio/wav",
                    )
            else:
                st.error("오디오 데이터를 처리할 수 없습니다.")
        else:
            st.warning("먼저 오디오를 녹음하세요.")

# 페이지 푸터
st.markdown("---")
st.markdown("© 2025 파동의 상쇄 간섭을 활용한 지능형 소음 제어 시스템")
