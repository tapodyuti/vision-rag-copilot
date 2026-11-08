import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Vision-RAG Safety Copilot", page_icon="🛡️", layout="wide")

st.title("🛡️ Vision-RAG Industrial Safety Copilot")
st.markdown("Analyze uploaded videos with object detection and search stored safety-event records.")

# Sidebar for actions
st.sidebar.header("Control Panel")
if st.sidebar.button("🚨 Simulate Camera Detection Event"):
    event_id = str(uuid.uuid4())[:8]
    # Send mock event to RAG service
    payload = {
        "event_id": event_id,
        "timestamp": "2026-10-17 14:30:00",
        "location": "Zone A - Warehouse",
        "description": "Worker detected operating heavy machinery without a safety helmet."
    }
    try:
        response = requests.post("http://rag-service:8002/add_event", json=payload)
        if response.status_code == 200:
            st.sidebar.success(f"Event {event_id} logged & vectorized!")
        else:
            st.sidebar.error("Failed to log event to RAG service.")
    except Exception as e:
        st.sidebar.error(f"Connection error: {e}")

# Main Layout: Two Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 Live Camera Feed & Vision Logs")
    st.info("Upload a video to scan it for objects, or use the demo camera trigger below.")
    
    # Fetch status from vision service
    try:
        v_res = requests.get("http://vision-service:8001/").json()
        st.write("Status:", v_res.get("status"))
    except:
        st.warning("Vision service offline or initializing...")

    st.subheader("Analyze an uploaded video")
    st.caption(
        "Supported formats: MP4, MOV, AVI, MKV, WebM, M4V, MPG, and MPEG. "
        "Codec support depends on the video. Uploads are limited to 200 MB, and the model samples about one frame per second."
    )
    st.caption(
        "This demo uses a general-object model, not a helmet or PPE detector. "
        "Do not treat its results as a safety-compliance decision."
    )
    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["avi", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "webm"],
        key="video_upload",
    )
    if uploaded_video:
        st.video(uploaded_video)

    if st.button("Analyze video", disabled=uploaded_video is None) and uploaded_video is not None:
        st.session_state.pop("video_analysis_result", None)
        st.session_state.pop("video_analysis_index_status", None)
        try:
            with st.spinner("Uploading and analyzing video..."):
                response = requests.post(
                    "http://vision-service:8001/analyze_video",
                    files={
                        "file": (
                            uploaded_video.name,
                            uploaded_video.getvalue(),
                            uploaded_video.type or "application/octet-stream",
                        )
                    },
                    timeout=(10, 1800),
                )
                response.raise_for_status()
                analysis = response.json()
                st.session_state["video_analysis_result"] = analysis
        except requests.HTTPError as error:
            detail = error.response.text if error.response is not None else str(error)
            st.error(f"Video analysis failed: {detail}")
        except requests.RequestException as error:
            st.error(f"Video analysis failed: {error}")
        else:
            detections_by_label = {}
            for detection in analysis.get("detections", []):
                label = detection.get("label")
                if isinstance(label, str):
                    detections_by_label.setdefault(label, []).append(detection)

            analysis_id = uuid.uuid4().hex
            events = []
            for index, (label, count) in enumerate(
                sorted(analysis.get("detections_by_class", {}).items())
            ):
                occurrences = detections_by_label.get(label, [])
                timestamps = ", ".join(
                    f"{detection['time_seconds']}s "
                    f"(confidence {detection['confidence']})"
                    for detection in occurrences[:20]
                )
                description = (
                    f"General-object video analysis of '{analysis['filename']}': "
                    f"detected {label} {count} times across sampled frames."
                )
                if timestamps:
                    description += f" Sampled occurrences include: {timestamps}."
                events.append({
                    "event_id": f"{analysis_id}-{index}",
                    "timestamp": "video analysis",
                    "location": analysis["filename"],
                    "description": description,
                })

            if not events:
                events.append({
                    "event_id": f"{analysis_id}-0",
                    "timestamp": "video analysis",
                    "location": analysis["filename"],
                    "description": (
                        f"General-object video analysis of '{analysis['filename']}': "
                        "no recognizable objects were detected in the sampled frames."
                    ),
                })

            try:
                with st.spinner("Adding video findings to searchable memory..."):
                    rag_response = requests.post(
                        "http://rag-service:8002/add_events",
                        json={"events": events},
                        timeout=(10, 180),
                    )
                    rag_response.raise_for_status()
                    stored_count = rag_response.json()["events_stored"]
                st.session_state["video_analysis_index_status"] = (
                    analysis["filename"],
                    f"Added {stored_count} video finding(s) to searchable memory.",
                    True,
                )
            except requests.HTTPError as error:
                detail = error.response.text if error.response is not None else str(error)
                st.session_state["video_analysis_index_status"] = (
                    analysis["filename"],
                    f"Video analyzed, but its findings could not be saved to search: {detail}",
                    False,
                )
            except requests.RequestException as error:
                st.session_state["video_analysis_index_status"] = (
                    analysis["filename"],
                    f"Video analyzed, but RAG search could not be reached: {error}",
                    False,
                )

    analysis = st.session_state.get("video_analysis_result")
    if analysis and uploaded_video and analysis.get("filename") == uploaded_video.name:
        index_status = st.session_state.get("video_analysis_index_status")
        if index_status and index_status[0] == analysis["filename"]:
            if index_status[2]:
                st.success(index_status[1])
            else:
                st.error(index_status[1])

        st.write(f"Results for **{analysis['filename']}**")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Duration", f"{analysis['duration_seconds']} s")
        metric2.metric("Frames sampled", analysis["sampled_frames"])
        metric3.metric("Objects found", analysis["total_detections"])

        if analysis["detections_by_class"]:
            st.write("Object counts")
            st.bar_chart(analysis["detections_by_class"])
            st.write("Sampled detections")
            st.dataframe(analysis["detections"], use_container_width=True)
            if analysis["detections_truncated"]:
                st.caption("Showing the first 500 detections.")
        else:
            st.info("No recognizable objects were detected in the sampled frames.")
        st.caption(analysis["model_note"])

    if st.button("Run Manual Detection Trigger"):
        try:
            det_res = requests.post("http://vision-service:8001/detect_stream").json()
            st.json(det_res)
        except Exception as e:
            st.error(f"Could not connect to Vision Service: {e}")

with col2:
    st.subheader("💬 Video Q&A and RAG Search")
    active_panel = st.radio(
        "Choose a panel",
        ["Ask about a video", "Search event records", "Search history"],
        horizontal=True,
        key="rag_active_panel",
        label_visibility="collapsed",
    )

    if active_panel == "Ask about a video":
        st.caption(
            "Ask follow-up questions about sampled frames and object detections. Audio "
            "isn't transcribed; answers may be incomplete and aren't PPE analysis."
        )
        st.info(
            "Automatic routing uses local Ollama with two sampled frames for "
            "straightforward questions. Complex questions may send up to 12 sampled "
            "frames to Google Gemini when cloud access is configured."
        )
        try:
            videos_response = requests.get(
                "http://vision-service:8001/videos",
                timeout=(10, 30),
            )
            videos_response.raise_for_status()
            videos = videos_response.json().get("videos", [])
        except requests.RequestException as error:
            videos = []
            st.error(f"Could not load analyzed videos: {error}")

        if videos:
            video_options = {
                f"{video['filename']} ({video['duration_seconds']} s, {video['video_id'][:8]})": video["video_id"]
                for video in videos
            }
            selected_video_label = st.selectbox(
                "Choose an analyzed video",
                list(video_options),
            )
            video_id = video_options[selected_video_label]
            chat_key = f"video_chat_{video_id}"
            messages = st.session_state.setdefault(chat_key, [])

            for message in messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("provider"):
                        st.caption(
                            f"Answered by {message['provider']} ({message['model']})."
                        )
                        if message["provider"] == "Gemini":
                            st.warning(
                                "This answer sent the question and sampled video frames "
                                "to Google Gemini."
                            )
                        if message.get("frame_timestamps"):
                            formatted_timestamps = ", ".join(
                                f"{timestamp}s"
                                for timestamp in message["frame_timestamps"]
                            )
                            st.caption(f"Frames considered at: {formatted_timestamps}")

            with st.form(f"video_chat_form_{video_id}", clear_on_submit=True):
                video_query = st.text_input(
                    "Your question",
                    placeholder="Ask a question about this video...",
                    key=f"video_question_{video_id}",
                )
                question_submitted = st.form_submit_button("Send")

            if question_submitted:
                if not video_query.strip():
                    st.info("Enter a question before sending.")
                    st.stop()
                history = [
                    {"role": message["role"], "content": message["content"]}
                    for message in messages[-8:]
                ]
                messages.append({"role": "user", "content": video_query})
                try:
                    with st.spinner("Checking video frames and preparing an answer..."):
                        response = requests.post(
                            "http://rag-service:8002/ask",
                            json={
                                "video_id": video_id,
                                "query": video_query,
                                "history": history,
                            },
                            timeout=(10, 360),
                        )
                        response.raise_for_status()
                        answer_result = response.json()
                    messages.append({
                        "role": "assistant",
                        "content": answer_result["answer"],
                        "provider": answer_result["provider"],
                        "model": answer_result["model"],
                        "frame_timestamps": answer_result.get("frame_timestamps", []),
                    })
                except requests.HTTPError as error:
                    detail = error.response.text if error.response is not None else str(error)
                    messages.append({
                        "role": "assistant",
                        "content": f"Could not answer the video question: {detail}",
                    })
                except requests.RequestException as error:
                    messages.append({
                        "role": "assistant",
                        "content": f"Could not reach the video Q&A service: {error}",
                    })
                except (KeyError, ValueError) as error:
                    messages.append({
                        "role": "assistant",
                        "content": f"The Q&A service returned an invalid response: {error}",
                    })
                st.rerun()
        else:
            st.info("Upload and analyze a video to ask questions about it.")

    elif active_panel == "Search event records":
        st.caption(
            "Search existing safety-event records and indexed general-object "
            "observations using semantic similarity."
        )
        with st.form("rag_search_form"):
            user_query = st.text_input(
                "Search stored records:",
                "Show me missing helmet violations",
                key="rag_search_query",
            )
            search_submitted = st.form_submit_button("Search Memory")

        if search_submitted:
            if user_query.strip():
                try:
                    response = requests.get(
                        "http://rag-service:8002/search",
                        params={"query": user_query},
                        timeout=(10, 60),
                    )
                    response.raise_for_status()
                    res = response.json()
                    st.session_state["latest_search_result"] = res
                except requests.RequestException as error:
                    st.error(f"Could not search RAG memory: {error}")
                except (KeyError, ValueError) as error:
                    st.error(f"The search service returned an invalid response: {error}")
            else:
                st.info("Enter a question before searching.")

        latest_search = st.session_state.get("latest_search_result")
        if latest_search:
            st.write(f"### Matched records for: {latest_search['query']}")
            result_groups = latest_search.get("matched_events") or []
            matches = result_groups[0] if result_groups else []
            matches = [
                match for match in matches
                if isinstance(match, str) and match.strip()
            ]
            if matches:
                for idx, match in enumerate(matches):
                    st.warning(f"Result {idx + 1}: {match}")
            else:
                st.info("No matching records found in memory.")

    else:
        st.caption(
            "Search history is saved across sessions. Select an entry to review its "
            "original results without rerunning the search."
        )
        try:
            history_response = requests.get(
                "http://rag-service:8002/search_history",
                params={"limit": 50},
                timeout=(10, 30),
            )
            history_response.raise_for_status()
            saved_searches = history_response.json().get("searches", [])
            if saved_searches:
                searches_by_id = {
                    search["id"]: search for search in saved_searches
                }
                selected_search_id = st.selectbox(
                    "Past searches (saved results)",
                    options=list(searches_by_id),
                    format_func=lambda search_id: (
                        f"{searches_by_id[search_id]['created_at']} — "
                        f"{searches_by_id[search_id]['query']}"
                    ),
                    key="saved_search_selection",
                )
                selected_search = searches_by_id[selected_search_id]
                st.write(f"### Results for: {selected_search['query']}")
                result_groups = selected_search.get("matched_events") or []
                matches = result_groups[0] if result_groups else []
                matches = [
                    match for match in matches
                    if isinstance(match, str) and match.strip()
                ]
                if matches:
                    for idx, match in enumerate(matches):
                        st.warning(f"Result {idx + 1}: {match}")
                else:
                    st.info("No matching records were found for this saved search.")
            else:
                st.info("Your searches will be saved here for future visits.")
        except requests.RequestException as error:
            st.error(f"Could not load saved searches: {error}")
        except (KeyError, ValueError) as error:
            st.error(f"The search-history service returned invalid data: {error}")